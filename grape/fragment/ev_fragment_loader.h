/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_
#define GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_

#include <mpi.h>

#include <algorithm>
#include <any>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/column_reader.h>
#include "arrow/filesystem/api.h"
#include "grape/fragment/basic_efile_fragment_loader.h"
#include "grape/fragment/basic_fragment_loader.h"
#include "grape/fragment/basic_local_fragment_loader.h"
#include "grape/fragment/basic_rb_fragment_loader.h"
#include "grape/io/line_parser_base.h"
#include "grape/io/local_io_adaptor.h"
#include "grape/io/tsv_line_parser.h"
#include "grape/worker/comm_spec.h"

int BATCH_SIZE = 1024*4;
namespace grape {

/**
 * @brief EVFragmentLoader is a loader to load fragments from separated
 * efile and vfile.
 *
 * @tparam FRAG_T Fragment type.
 * @tparam IOADAPTOR_T IOAdaptor type.
 * @tparam LINE_PARSER_T LineParser type.
 */
template <typename FRAG_T, typename IOADAPTOR_T = LocalIOAdaptor,
          typename LINE_PARSER_T =
              TSVLineParser<typename FRAG_T::oid_t, typename FRAG_T::vdata_t,
                            typename FRAG_T::edata_t>>
class EVFragmentLoader {
  using fragment_t = FRAG_T;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename fragment_t::vid_t;
  using vdata_t = typename fragment_t::vdata_t;
  using edata_t = typename fragment_t::edata_t;

  using io_adaptor_t = IOADAPTOR_T;
  using line_parser_t = LINE_PARSER_T;

  static constexpr LoadStrategy load_strategy = fragment_t::load_strategy;

  static_assert(std::is_base_of<LineParserBase<oid_t, vdata_t, edata_t>,
                                LINE_PARSER_T>::value,
                "LineParser type is invalid");

 public:
  explicit EVFragmentLoader(const CommSpec& comm_spec)
      : comm_spec_(comm_spec), basic_fragment_loader_(nullptr) {}

  ~EVFragmentLoader() = default;

  std::vector<int64_t> setParquetPartialReadImpl(int total_lines,
                                                 int total_parts) {
    int64_t part_size = total_lines / total_parts;
    std::vector<int64_t> partial_read_offset;
    partial_read_offset.resize(total_parts + 1, 0);
    partial_read_offset[total_parts] = total_lines;

    // move breakpoint to the next of nearest character '\n'
    for (int i = 1; i < total_parts; ++i) {
      partial_read_offset[i] = i * part_size;
    }
    return partial_read_offset;
  }
  void readWithChunkReader() {
    // read from count file
    auto path =
        "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/path/"
        "ordered_by_source/edge_count0";
    auto fs = arrow::fs::FileSystemFromUriOrPath(path).ValueOrDie();
    std::shared_ptr<arrow::io::InputStream> input =
        fs->OpenInputStream(path).ValueOrDie();
    auto edge_num = input->Read(sizeof(int64_t)).ValueOrDie();
    int64_t* edge_num_ptr = (int64_t*) edge_num->data();
    auto partial_read_offset =
        setParquetPartialReadImpl(*edge_num_ptr, comm_spec_.worker_num());
    if (comm_spec_.worker_id() == 0) {
      std::cout << "edge_num:" << *edge_num_ptr << std::endl;
    }
    // std::cout << "loading edges from parquet files..." << std::endl;
    std::shared_ptr<arrow::io::ReadableFile> edgeFile =
        arrow::io::ReadableFile::Open(
            "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/"
            "path/"
            "ordered_by_source/adj_list/part0/chunk0")
            .ValueOrDie();
    std::shared_ptr<arrow::io::ReadableFile> weightFile =
        arrow::io::ReadableFile::Open(
            "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/"
            "path/ordered_by_source/weight/part0/chunk0")
            .ValueOrDie();
    std::unique_ptr<parquet::arrow::FileReader> edgeReader;
    std::unique_ptr<parquet::arrow::FileReader> weightReader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(
        edgeFile, arrow::default_memory_pool(), &edgeReader));
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(
        weightFile, arrow::default_memory_pool(), &weightReader));

    std::shared_ptr<arrow::Table> edgeTable;
    std::shared_ptr<arrow::Table> weightTable;

    PARQUET_THROW_NOT_OK(edgeReader->ReadTable(&edgeTable));
    PARQUET_THROW_NOT_OK(weightReader->ReadTable(&weightTable));
    auto src_col_index = edgeTable->schema()->GetFieldIndex("_graphArSrcIndex");
    auto dst_col_index = edgeTable->schema()->GetFieldIndex("_graphArDstIndex");
    auto weight_col_index = weightTable->schema()->GetFieldIndex("weight");
    int64_t row_offset = 0;
    int index = comm_spec_.worker_id();
    for (int64_t chunk_idx = 0;
         chunk_idx < edgeTable->column(src_col_index)->num_chunks();
         ++chunk_idx) {
      auto src_chunk = edgeTable->column(src_col_index)->chunk(chunk_idx);
      if (row_offset + src_chunk->length() < partial_read_offset[index]) {
        continue;
      }
      auto src_column = std::static_pointer_cast<arrow::Int64Array>(src_chunk);
      auto dst_chunk = edgeTable->column(dst_col_index)->chunk(chunk_idx);
      auto dst_column = std::static_pointer_cast<arrow::Int64Array>(dst_chunk);
      auto weight_chunk =
          weightTable->column(weight_col_index)->chunk(chunk_idx);
      auto weight_column =
          std::static_pointer_cast<arrow::Int64Array>(weight_chunk);
      int64_t start =
          std::max(partial_read_offset[index] - row_offset, (int64_t) 0);
      for (int64_t row = start; row < src_column->length(); ++row) {
        if (src_column->IsValid(row)) {
          if (row_offset + row >= partial_read_offset[index + 1]) {
            break;
          }
          int64_t src = src_column->GetView(row);
          int64_t dst = dst_column->GetView(row);
          int64_t weigth = weight_column->GetView(row);
          double edge_data = weigth * 1.0;
          basic_fragment_loader_->AddEdge(src, dst, edge_data);
        }
      }
      row_offset +=
          src_column->length();  // Update the row offset for the next chunk
      if (row_offset >= partial_read_offset[index + 1]) {
        break;
      }
    }
  }
  void readWithRowGroupReader() {
    int index = comm_spec_.worker_id();
    // read from count file
    auto path =
        "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/path/"
        "ordered_by_source/edge_count0";
    auto fs = arrow::fs::FileSystemFromUriOrPath(path).ValueOrDie();
    std::shared_ptr<arrow::io::InputStream> input =
        fs->OpenInputStream(path).ValueOrDie();
    auto edge_num = input->Read(sizeof(int64_t)).ValueOrDie();
    int64_t* edge_num_ptr = (int64_t*) edge_num->data();
    auto partial_read_offset =
        setParquetPartialReadImpl(*edge_num_ptr, comm_spec_.worker_num());
    if (comm_spec_.worker_id() == 0) {
      std::cout << "edge_num:" << *edge_num_ptr << std::endl;
    }
    std::unique_ptr<parquet::ParquetFileReader> edge_parquet_reader =
        parquet::ParquetFileReader::OpenFile(
            "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/"
            "path/"
            "ordered_by_source/adj_list/part0/chunk0",
            false);
    std::unique_ptr<parquet::ParquetFileReader> property_parquet_reader =
        parquet::ParquetFileReader::OpenFile(
            "/Users/yangxk/code/apache/libgrape-lite/dataset/graphar/edge/path/"
            "ordered_by_source/weight/part0/chunk0",
            false);
    auto weight_index =
        property_parquet_reader->metadata()->schema()->ColumnIndex("weight");
    auto meta_data = edge_parquet_reader->metadata();
    auto num_rg = meta_data->num_row_groups();
    int offset = 0;
    int64_t* srcs = new int64_t[BATCH_SIZE];
    int64_t* dsts = new int64_t[BATCH_SIZE];
    int64_t* weights = new int64_t[BATCH_SIZE];
    for (int rg = 0; rg < num_rg; ++rg) {
      auto rg_num_rows =
          edge_parquet_reader->metadata()->RowGroup(rg)->num_rows();
      if (offset + rg_num_rows <= partial_read_offset[index]) {
        offset += rg_num_rows;
        continue;
      }
      if (offset >= partial_read_offset[index + 1]) {
        return;
      }
      // Get the RowGroup Reader
      std::shared_ptr<parquet::RowGroupReader> edge_row_group_reader =
          edge_parquet_reader->RowGroup(rg);
      std::shared_ptr<parquet::RowGroupReader> property_row_group_reader =
          property_parquet_reader->RowGroup(rg);

      int64_t values_read = 0;
      int64_t rows_read = 0;

      ARROW_UNUSED(rows_read);  // prevent warning in release build
      auto src_col_reader = edge_row_group_reader->Column(0);
      auto dst_col_reader = edge_row_group_reader->Column(1);
      auto weight_col_reader = property_row_group_reader->Column(weight_index);
      parquet::Int64Reader* src_reader =
          static_cast<parquet::Int64Reader*>(src_col_reader.get());
      parquet::Int64Reader* dst_reader =
          static_cast<parquet::Int64Reader*>(dst_col_reader.get());
      parquet::Int64Reader* weight_reader =
          static_cast<parquet::Int64Reader*>(weight_col_reader.get());
      // 由于前面的都是BATCH_SIZE，所以可以直接跳过去
      // TODO 这里假设存储的类型都为int64，不然不能这样跳
      if (partial_read_offset[index] > offset) {
        src_reader->Skip(partial_read_offset[index] - offset);
        dst_reader->Skip(partial_read_offset[index] - offset);
        weight_reader->Skip(partial_read_offset[index] - offset);
        offset = partial_read_offset[index];
      }
      // Read all the rows in the column
      while (src_reader->HasNext()) {
        // Read BATCH_SIZE values at a time. The number of rows read is
        // returned. values_read contains the number of non-null rows
        auto src_rows_read = src_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr,
                                                   srcs, &values_read);
        auto dst_rows_read = dst_reader->ReadBatch(BATCH_SIZE, nullptr, nullptr,
                                                   dsts, &values_read);
        auto weight_rows_read = weight_reader->ReadBatch(
            BATCH_SIZE, nullptr, nullptr, weights, &values_read);
        assert(src_rows_read == dst_rows_read &&
               src_rows_read == weight_rows_read);
        for (int64_t i = 0; i < src_rows_read; ++i) {
          if (offset >= partial_read_offset[index + 1]) {
            return;
          }
          basic_fragment_loader_->AddEdge(srcs[i], dsts[i],
                                          (double) weights[i]);
          offset++;
        }
        if (offset >= partial_read_offset[index + 1]) {
          return;
          ;
        }
      }
    }
  }

  std::shared_ptr<fragment_t> LoadFragment(const std::string& efile,
                                           const std::string& vfile,
                                           const LoadGraphSpec& spec) {
    std::shared_ptr<fragment_t> fragment(nullptr);
    if (spec.deserialize) {
      bool deserialized = DeserializeFragment<fragment_t, IOADAPTOR_T>(
          fragment, comm_spec_, efile, vfile, spec);
      int flag = 0;
      int sum = 0;
      if (!deserialized) {
        flag = 1;
      }
      MPI_Allreduce(&flag, &sum, 1, MPI_INT, MPI_SUM, comm_spec_.comm());
      if (sum != 0) {
        fragment.reset();
        if (comm_spec_.worker_id() == 0) {
          VLOG(2) << "Deserialization failed, start loading graph from "
                     "efile and vfile.";
        }
      } else {
        return fragment;
      }
    }

    if (vfile.empty()) {
      basic_fragment_loader_ =
          std::unique_ptr<BasicEFileFragmentLoader<fragment_t>>(
              new BasicEFileFragmentLoader<fragment_t>(comm_spec_, spec));
    } else {
      if (spec.idxer_type != IdxerType::kLocalIdxer) {
        if (spec.rebalance) {
          basic_fragment_loader_ =
              std::unique_ptr<BasicRbFragmentLoader<fragment_t>>(
                  new BasicRbFragmentLoader<fragment_t>(comm_spec_, spec));
        } else {
          basic_fragment_loader_ =
              std::unique_ptr<BasicFragmentLoader<fragment_t>>(
                  new BasicFragmentLoader<fragment_t>(comm_spec_, spec));
        }
      } else {
        basic_fragment_loader_ =
            std::unique_ptr<BasicLocalFragmentLoader<fragment_t>>(
                new BasicLocalFragmentLoader<fragment_t>(comm_spec_, spec));
      }
    }

    if (!vfile.empty()) {
      MPI_Barrier(comm_spec_.comm());
      double t0 = -grape::GetCurrentTime();

      auto io_adaptor = std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(vfile));
      io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                 comm_spec_.worker_num());
      io_adaptor->Open();
      std::string line;
      vdata_t v_data;
      oid_t vertex_id;
      size_t line_no = 0;
      while (io_adaptor->ReadLine(line)) {
        ++line_no;
        if (line_no % 1000000 == 0) {
          VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][vfile] "
                   << line_no;
        }
        if (line.empty() || line[0] == '#')
          continue;
        try {
          line_parser_.LineParserForVFile(line, vertex_id, v_data);
        } catch (std::exception& e) {
          VLOG(1) << e.what();
          continue;
        }
        basic_fragment_loader_->AddVertex(vertex_id, v_data);
      }
      io_adaptor->Close();

      MPI_Barrier(comm_spec_.comm());
      t0 += grape::GetCurrentTime();
      if (comm_spec_.worker_id() == 0) {
        VLOG(1) << "finished reading vertices inputs, time: " << t0 << " s";
      }

      double t1 = -grape::GetCurrentTime();
      basic_fragment_loader_->ConstructVertices();

      MPI_Barrier(comm_spec_.comm());
      t1 += grape::GetCurrentTime();
      if (comm_spec_.worker_id() == 0) {
        VLOG(1) << "finished constructing vertices, time: " << t1 << " s";
      }
    } else {
      basic_fragment_loader_->ConstructVertices();
    }

    double t2 = -grape::GetCurrentTime();
    {
      if constexpr (std::is_same<edata_t, double>::value &&
                    (std::is_same<oid_t, int64_t>::value ||
                     std::is_same<oid_t, int32_t>::value)) {
        readWithRowGroupReader();
      } else {
        auto io_adaptor =
            std::unique_ptr<IOADAPTOR_T>(new IOADAPTOR_T(std::string(efile)));
        io_adaptor->SetPartialRead(comm_spec_.worker_id(),
                                   comm_spec_.worker_num());
        io_adaptor->Open();
        std::string line;
        edata_t e_data;
        oid_t src, dst;

        size_t lineNo = 0;
        while (io_adaptor->ReadLine(line)) {
          ++lineNo;
          if (lineNo % 1000000 == 0) {
            VLOG(10) << "[worker-" << comm_spec_.worker_id() << "][efile] "
                     << lineNo;
          }
          if (line.empty() || line[0] == '#')
            continue;

          try {
            line_parser_.LineParserForEFile(line, src, dst, e_data);
          } catch (std::exception& e) {
            VLOG(1) << e.what();
            continue;
          }

          basic_fragment_loader_->AddEdge(src, dst, e_data);
        }
        io_adaptor->Close();
      }
    }
    MPI_Barrier(comm_spec_.comm());
    t2 += grape::GetCurrentTime();
    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished reading edges inputs, time: " << t2 << " s";
    }

    double t3 = -grape::GetCurrentTime();
    basic_fragment_loader_->ConstructFragment(fragment);
    MPI_Barrier(comm_spec_.comm());
    t3 += grape::GetCurrentTime();

    if (comm_spec_.worker_id() == 0) {
      VLOG(1) << "finished constructing fragment, time: " << t3 << " s";
    }

    if (spec.serialize) {
      bool serialized = SerializeFragment<fragment_t, IOADAPTOR_T>(
          fragment, comm_spec_, efile, vfile, spec);
      if (!serialized) {
        VLOG(2) << "[worker-" << comm_spec_.worker_id()
                << "] Serialization failed.";
      }
    }

    return fragment;
  }

 private:
  CommSpec comm_spec_;

  std::unique_ptr<BasicFragmentLoaderBase<fragment_t>> basic_fragment_loader_;
  line_parser_t line_parser_;
};

}  // namespace grape

#endif  // GRAPE_FRAGMENT_EV_FRAGMENT_LOADER_H_
