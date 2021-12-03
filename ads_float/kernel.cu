/*  -*- mode: c++ -*-  */
#include <cuda.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>
#include <cassert>
#include "../cub-1.8.0/cub/cub.cuh"
#include "common.h"
#include "csr_graph.h"
#include "support.h"
#include "wl.h"

#define TB_SIZE 768
int CUDA_DEVICE = 0;
int start_node = 0;
char *INPUT, *OUTPUT;

__global__ void kernel(CSRGraph graph, int src) {
	unsigned tid = thread_id_x() + block_id_x() * block_dim_x();
	unsigned nthreads = block_dim_x() * grid_dim_x();

	index_type node_end;
	node_end = (graph).nnodes;
	for (index_type node = 0 + tid; node < node_end; node += nthreads) {
		graph.node_data[node] = (node == src) ? 0 : INF;
	}
}
#define HOR_EDGE 16
__global__ void
__launch_bounds__(896, 1)
sssp_kernel(CSRGraph graph, worklist wl, unsigned* work_count) {
	wl.init_regular();

	unsigned tid = thread_id_x() + block_id_x() * block_dim_x();
	const int warpid = thread_id_x() / 32;
	unsigned total_work = 0;
	__shared__ unsigned char leader_lane_tb_storage[TB_SIZE * 32];
	unsigned char * leader_lane_storage = &(leader_lane_tb_storage[warpid * 32 * 32]);

	unsigned tb_coop_threshold = max(32 * TB_COOP_MUL, (graph.nedges / graph.nnodes) * TB_COOP_MUL);

	while (1) {

		//get work
		unsigned long long m_assignment;
		//this does tb coop processing also
		unsigned num = wl.get_assignment(graph, m_assignment, warpid, work_count, total_work);
		unsigned ptr = agm_get_real_ptr(m_assignment);
		unsigned src_bag_id = agm_get_bag_id(m_assignment);
		uint wl_offset = get_lane_id();
		int m_first_edge;
		int m_size = 0;
		int m_vertex;
		if (wl_offset < num) {
			m_vertex = wl.pop_work(ptr + wl_offset);
			if (m_vertex < graph.nnodes) {
				m_first_edge = graph.row_start[m_vertex];
				m_size = graph.row_start[m_vertex + 1] - m_first_edge;
				total_work++;
			}
		}

		//do tb coop assign, assign just one if multiple
		unsigned process_mask = __ballot_sync(FULL_MASK, m_size >= tb_coop_threshold);
		unsigned tb_coop_lane = find_ms_bit(process_mask);
		if (tb_coop_lane != NOT_FOUND) {
			wl.tb_coop_assign(m_vertex, m_first_edge, m_assignment, m_size, tb_coop_lane);
		}

		//do high degree vertex
		process_mask = __ballot_sync(FULL_MASK, m_size >= HOR_EDGE);
		while (process_mask != 0) {
			unsigned leader = find_ms_bit(process_mask);
			int leader_first_edge = __shfl_sync(FULL_MASK, m_first_edge, leader);
			int leader_size = __shfl_sync(FULL_MASK, m_size, leader);
			int leader_vertex = __shfl_sync(FULL_MASK, m_vertex, leader);
			for (int offset = get_lane_id(); offset < leader_size; offset += 32) {
				index_type edge = leader_first_edge + offset;
				index_type dst = graph.edge_dst[edge];
				edge_data_type wt = graph.edge_data[edge];
				node_data_type new_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[leader_vertex])) + wt;
				node_data_type dst_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[dst]));
				if (dst_dist > new_dist) {
					if (atomicMin_float(&(graph.node_data[dst]), new_dist) > new_dist) {
						unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, new_dist);
						wl.push_work(dst_bag_id, dst);
					}
				}
			}
			process_mask = set_bits(process_mask, 0, leader, 1);
		}
		if (m_size >= HOR_EDGE) {
			m_size = 0;
		}

		//do low degree vertex
		__syncwarp();
		int warp_offset;
		int warp_total;
		typedef cub::WarpScan<int> WarpScan;
		__shared__ typename WarpScan::TempStorage temp_storage[TB_SIZE / 32];
		WarpScan(temp_storage[warpid]).ExclusiveSum(m_size, warp_offset, warp_total);

		//write the target lane storage
		int write_idx = warp_offset;
		int write_end = warp_offset + m_size;
		while (write_idx < write_end) {
			leader_lane_storage[write_idx] = (unsigned char) get_lane_id();
			write_idx++;
		}
		__syncwarp();
		for (int read_idx_start = 0; read_idx_start < warp_total; read_idx_start += 32) {
			int m_read_idx = read_idx_start + get_lane_id();
			int leader = (int) leader_lane_storage[m_read_idx];
			int leader_warp_offset = __shfl_sync(FULL_MASK, warp_offset, leader);
			int leader_first_edge = __shfl_sync(FULL_MASK, m_first_edge, leader);
			int leader_vertex = __shfl_sync(FULL_MASK, m_vertex, leader);
			if (m_read_idx < warp_total) {
				int offset = m_read_idx - leader_warp_offset;
				index_type edge = leader_first_edge + offset;
				index_type dst = graph.edge_dst[edge];
				edge_data_type wt = graph.edge_data[edge];
				node_data_type new_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[leader_vertex])) + wt;
				node_data_type dst_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[dst]));
				if (dst_dist > new_dist) {
					if (atomicMin_float(&(graph.node_data[dst]), new_dist) > new_dist) {
						unsigned dst_bag_id = wl.dist_to_bag_id_int(src_bag_id, new_dist);
						wl.push_work(dst_bag_id, dst);
					}
				}
			}
		}
		__syncwarp();
		//free
		wl.epilog(m_assignment, num, false);
	}
}

__global__ void driver_kernel(int num_tb, int num_threads, worklist wl, CSRGraph gg, uint start_node, unsigned* work_count) {

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	wl_kernel<<<1, 192, 0, s1>>>(wl, start_node);

	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	sssp_kernel<<<num_tb, num_threads, 0, s2>>>(gg, wl, work_count);

}

void gg_main_pipe_1_wrapper(CSRGraph& hg, CSRGraph& gg, uint start_node, int num_tb, int num_threads, worklist wl, unsigned* work_count) {
	// gg_main_pipe_1_gpu<<<1,1>>>(gg,glevel,curdelta,i,DELTA,remove_dups_barrier,remove_dups_blocks,pipe,blocks,threads,cl_curdelta,cl_i, enable_lb);
	//gg_cg_gb<<<gg_main_pipe_1_gpu_gb_blocks, __tb_gg_main_pipe_1_gpu_gb>>>(
	//		gg, glevel, curdelta, i, DELTA, pipe, cl_curdelta, cl_i, enable_lb);

	driver_kernel<<<1, 1>>>(num_tb, num_threads, wl, gg, start_node, work_count);
	cudaDeviceSynchronize();
}

#define I_TIME 4
#define N_TIME 2
__global__ void profiler_kernel(CSRGraph gg, float* bk_wt, int warp_edge_interval) {

	int tid = thread_id_x() + block_id_x() * block_dim_x();
	int warp_id = tid / 32;
	int num_warp = block_dim_x() * grid_dim_x() / 32;

	typedef cub::BlockReduce<float, 1024> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	float thread_data[I_TIME];
	float tb_ave = 0;
	//find ave weight
	tb_ave = 0;
	for (int n = 0; n < N_TIME; n++) {
		for (int i = 0; i < I_TIME; i++) {
			unsigned warp_offset = ((n * I_TIME + i) * num_warp + warp_id) * warp_edge_interval;
			unsigned edge_id = warp_offset + cub::LaneId();
			if (edge_id < gg.nedges) {
				thread_data[i] = (float) gg.edge_data[edge_id];
			} else {
				thread_data[i] = 0;
			}
		}
		//do a reduction
		float sum = BlockReduce(temp_storage).Sum(thread_data);
		tb_ave += (sum / 1024 / I_TIME);
	}
	tb_ave = tb_ave / N_TIME;
	//store it back
	if (threadIdx.x == 0) {
		bk_wt[blockIdx.x] = tb_ave;
	}
}

void gg_main(CSRGraph& hg, CSRGraph& gg) {

	struct cudaDeviceProp dev_prop;
	cudaGetDeviceProperties(&dev_prop, CUDA_DEVICE);

	int num_threads = TB_SIZE;

	int num_tb_per_sm;
	int max_num_threads;
	int min_grid_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &max_num_threads, sssp_kernel);
	if (num_threads > max_num_threads) {
		printf("error max threads is %d, specified is %d\n", max_num_threads, num_threads);
		fflush(0);
		exit(1);
	}
	printf("max_num_threads is %d\n", max_num_threads);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_tb_per_sm, sssp_kernel, num_threads, 0);
	int num_sm = dev_prop.multiProcessorCount;
	int num_tb = num_tb_per_sm * (num_sm - 1);
	int num_warps = num_threads / 32;
	printf("sm count %u, tb per sm %u, tb size %u, num warp %u, total tb %u\n", num_sm, num_tb_per_sm, num_threads, num_warps, num_tb);

#define WL_SIZE_MUL 1.5f
	unsigned suggest_size = (unsigned)((float)hg.nedges * WL_SIZE_MUL) + (NUM_BAG * BLOCK_SIZE * 8);
	suggest_size = min(suggest_size, 536870912);
	unsigned* work_count;
	cudaMalloc((void **) &work_count, num_tb * num_threads * sizeof(unsigned));

#define RUN_LOOP 8


	worklist wl;
	wl.alloc(num_tb, num_warps, suggest_size, hg.nnodes, hg.nedges);

	unsigned long long agg_total_work = 0;
	float agg_time = 0;
	for (int loop = 0; loop < RUN_LOOP; loop++) {
		int other_num_tb = num_tb_per_sm * num_sm;

		kernel<<<other_num_tb, 1024>>>(gg, start_node);
		cudaMemset(work_count, 0, num_tb * num_threads * sizeof(unsigned));
		wl.reinit();
		cudaDeviceSynchronize();

		float elapsed_time;   // timing variables
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);

		float ave_degree = (float) hg.nedges / (float) hg.nnodes;
		float ave_wt;
		{
			//find delta
			int total_warp = other_num_tb * 1024 / 32;
			int warp_edge_interval = hg.nedges / (total_warp * I_TIME * N_TIME);
			float* host_bk_wt = (float*) malloc(other_num_tb * sizeof(float));
			float* bk_wt;
			cudaMalloc((void **) &bk_wt, other_num_tb * sizeof(float));
			profiler_kernel<<<other_num_tb, 1024>>>(gg, bk_wt, warp_edge_interval);
			cudaMemcpy(host_bk_wt, bk_wt, other_num_tb * sizeof(float), cudaMemcpyDeviceToHost);

			ave_wt = 0;
			for (int i = 0; i < other_num_tb; i++) {
				ave_wt += host_bk_wt[i];
			}
			ave_wt /= other_num_tb;
		}
		wl.set_param(ave_wt, ave_degree);
		gg_main_pipe_1_wrapper(hg, gg, start_node, num_tb, num_threads, wl, work_count);

		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);

		unsigned* work_count_host = (unsigned*) malloc(num_tb * num_threads * sizeof(unsigned));

		cudaMemcpy(work_count_host, work_count, num_tb * num_threads * sizeof(unsigned), cudaMemcpyDeviceToHost);
		unsigned long long total_work = 0;
		for (int i = 0; i < num_tb * num_threads; i++) {
			total_work += work_count_host[i];
		}

		agg_time += elapsed_time;
		agg_total_work += total_work;

		printf("%s Measured time for sample = %.3fs\n", hg.file_name, elapsed_time / 1000.0f);
		printf("total work is %llu\n", total_work);

	}

	FILE * d3;
	d3 = fopen("/dev/fd/3", "a");
	float ave_time = agg_time / RUN_LOOP;
	long long unsigned ave_work = agg_total_work / RUN_LOOP;
	fprintf(d3, "%s %.3f %llu\n", hg.file_name, ave_time, ave_work);
	wl.free();
	cudaFree(work_count);
	fclose(d3);

//clean up
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		usage(argc, argv);
		exit(1);
	}
	parse_args(argc, argv);
	cudaSetDevice(CUDA_DEVICE);
	CSRGraphTy g, gg;
	g.read(INPUT);
	g.copy_to_gpu(gg);
	gg_main(g, gg);
	gg.copy_to_cpu(g);
	output(g, OUTPUT);

	return 0;
}
