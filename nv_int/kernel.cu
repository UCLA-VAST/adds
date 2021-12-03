/*  -*- mode: c++ -*-  */
#include <cuda.h>
#include <inttypes.h>
#include <bitset>
#include <cmath>
#include <cassert>
#include "csr_graph.h"
#include "support.h"

#include <cuda_runtime.h>
#include <nvgraph.h>
typedef int * gint_p;
void check_status(nvgraphStatus_t status) {
	if ((int) status != 0) {
		printf("ERROR : %d\n", status);
		exit(0);
	}
}

int CUDA_DEVICE = 0;
char *INPUT, *OUTPUT;
int start_node;
void gg_main(CSRGraph& hg, CSRGraph& gg) {

	// nvgraph variables
	nvgraphStatus_t status;
	nvgraphHandle_t handle;
	nvgraphGraphDescr_t src_csr_graph;
	nvgraphCSRTopology32I_t CSR_input;
	cudaDataType_t edge_dimT = CUDA_R_32F;
	cudaDataType_t vertex_dimT = CUDA_R_32F;

	//edge list ptr
	int* src_offsets_h = (int*) hg.row_start;
	//edge list
	int* dst_indices_h = (int*) hg.edge_dst;
	//edge data
	float* weights_h = (float*) malloc(hg.nedges * sizeof(float));
	for (int i = 0; i < hg.nedges; i++) {
		weights_h[i] = (float) hg.edge_data[i];
	}
	//the CSC descriptor
	CSR_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));

	check_status(nvgraphCreate(&handle));
	check_status(nvgraphCreateGraphDescr(handle, &src_csr_graph));

	CSR_input->nvertices = hg.nnodes;
	CSR_input->nedges = hg.nedges;
	CSR_input->source_offsets = src_offsets_h;
	CSR_input->destination_indices = dst_indices_h;

	// Set graph connectivity and properties (tranfers)
	check_status(nvgraphSetGraphStructure(handle, src_csr_graph, (void*) CSR_input, NVGRAPH_CSR_32));
	check_status(nvgraphAllocateVertexData(handle, src_csr_graph, 1, &vertex_dimT));
	check_status(nvgraphAllocateEdgeData(handle, src_csr_graph, 1, &edge_dimT));
	check_status(nvgraphSetEdgeData(handle, src_csr_graph, (void*) weights_h, 0));

	// Convert to CSR graph
	nvgraphGraphDescr_t graph;
	check_status(nvgraphCreateGraphDescr(handle, &graph));
	check_status(nvgraphConvertGraph(handle, src_csr_graph, graph, NVGRAPH_CSC_32));

	// Solve
	int source_vert = start_node;

#define RUN_LOOP 8
	float agg_time = 0;
	for (int loop = 0; loop < RUN_LOOP; loop++) {
		float elapsed_time;   // timing variables
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);

		check_status(nvgraphSssp(handle, graph, 0, &source_vert, 0));

		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);
		agg_time += elapsed_time;
	}

	//output for checking

	//node data
	float* sssp_h = (float*) malloc(hg.nnodes * sizeof(float));
	check_status(nvgraphGetVertexData(handle, graph, (void*) sssp_h, 0));
	char file_name[256] = "nv_int_final_dist/";
	strcat(file_name,hg.file_name);
	FILE *fp = fopen(file_name, "w");
	for (int i = 0; i < hg.nnodes; i++) {
		float dist = sssp_h[i];
		if (dist == FLT_MAX) {
			fprintf(fp, "%d INF\n", i);
		} else {
			fprintf(fp, "%d %d\n", i, (int) dist);
		}
	}

	FILE * d3;
	d3 = fopen("/dev/fd/3", "a");
	fprintf(d3, "%s %.3f\n", hg.file_name, agg_time / RUN_LOOP);
	fclose(d3);
	return;

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

	return 0;
}
