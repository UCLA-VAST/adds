/*
 * gb_comm.h
 *
 *  Created on: May 29, 2018
 *      Author: redudie
 */

#ifndef WL_H_
#define WL_H_

#define PRINT_DEBUG 0
#define PRINT_BAG 0

#define NUM_BAG 32
#define MAX_CON_BAG 4
#define SEG_SIZE 256

#define EXIT ((unsigned long long)0xffffffffffffffff)
#define BLOCK_SIZE (64*1024)
#define BIT_CACHE_SIZE (2048)
#define PRE_ALLOC_DIST (4)
#define PREFETCH_DIST (1)
#define MAX_TB 128
#define INIT_FACTOR 32
#define FIXED_SHIFT 9

inline static __device__ __host__ int roundup(int a, int r) {
	return ((a + r - 1) / r) * r;
}

__device__ __forceinline__ unsigned agm_get_low_ptr(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 0, 16);
}
__device__ __forceinline__ unsigned agm_get_real_bk_idx(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 16, 16);
}
__device__ __forceinline__ unsigned agm_get_real_ptr(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 0, 32);
}
__device__ __forceinline__ unsigned agm_get_size(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 32, 16);
}
__device__ __forceinline__ unsigned agm_get_grain_bit(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 55, 3);
}
__device__ __forceinline__ unsigned agm_get_bag_id(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 58, 5);
}
__device__ __forceinline__ unsigned agm_get_seq(unsigned long long assginment) {
	return (unsigned) extract_bits_64(assginment, 63, 1);
}

__device__ __forceinline__ unsigned long long agm_set_low_ptr(unsigned long long assginment, unsigned low_ptr) {
	return set_bits_64(assginment, low_ptr, 0, 16);
}
__device__ __forceinline__ unsigned long long agm_set_real_bk_idx(unsigned long long assginment, unsigned real_bk_idx) {
	return set_bits_64(assginment, real_bk_idx, 16, 16);
}
__device__ __forceinline__ unsigned long long agm_set_real_ptr(unsigned long long assginment, unsigned real_ptr) {
	return set_bits_64(assginment, real_ptr, 0, 32);
}
__device__ __forceinline__ unsigned long long agm_set_size(unsigned long long assginment, unsigned size) {
	return set_bits_64(assginment, size, 32, 16);
}
__device__ __forceinline__ unsigned long long agm_set_grain_bit(unsigned long long assginment, unsigned grain) {
	return set_bits_64(assginment, grain, 55, 3);
}
__device__ __forceinline__ unsigned long long agm_set_bag_id(unsigned long long assginment, unsigned bag_id) {
	return set_bits_64(assginment, bag_id, 58, 5);
}
__device__ __forceinline__ unsigned long long agm_set_seq(unsigned long long assginment, unsigned seq) {
	return set_bits_64(assginment, seq, 63, 1);
}

#define BFG_DIST_TSH_BITS 20

__device__ __forceinline__ unsigned bfg_get_delta(unsigned delta_info) {
	return extract_bits(delta_info, 27, 5);
}

__device__ __forceinline__ unsigned bfg_get_bag(unsigned delta_info) {
	return extract_bits(delta_info, 22, 5);
}
__device__ __forceinline__ unsigned bfg_get_dist(unsigned delta_info) {
	return extract_bits(delta_info, 0, 22);
}

__device__ __forceinline__ unsigned bfg_set_delta(unsigned delta_info, unsigned delta) {
	return set_bits(delta_info, delta, 27, 5);
}

__device__ __forceinline__ unsigned bfg_set_bag(unsigned delta_info, unsigned bag) {
	return set_bits(delta_info, bag, 22, 5);
}
__device__ __forceinline__ unsigned bfg_set_dist(unsigned delta_info, unsigned dist) {
	return set_bits(delta_info, dist, 0, 22);
}

__host__ __device__ __forceinline__ unsigned r_up(unsigned a, unsigned r) {
	return ((a + r - 1) / r) * r;
}

#define EDGE_WRONG_LEVEL 32
#define MAX_EDGE_WRONG (EDGE_WRONG_LEVEL*2)

__device__ __forceinline__ unsigned rs_get_seq(unsigned reg_stat) {
	return extract_bits(reg_stat, 31, 1);
}
__device__ __forceinline__ float rs_get_per_assign_wrong(unsigned reg_stat) {
	unsigned wrong = extract_bits(reg_stat, 0, 6);
	return (float) wrong / (float) EDGE_WRONG_LEVEL;
}
__device__ __forceinline__ unsigned rs_get_processing(unsigned reg_stat) {
	return extract_bits(reg_stat, 6, 11);
}
__device__ __forceinline__ unsigned rs_get_assigned(unsigned reg_stat) {
	return extract_bits(reg_stat, 17, 14);
}
__device__ __forceinline__ unsigned rs_set_seq(unsigned reg_stat, unsigned seq) {
	return set_bits(reg_stat, seq, 31, 1);
}
__device__ __forceinline__ unsigned rs_set_per_assign_wrong(unsigned reg_stat, float per_assign_wrong) {
	unsigned wrong = __float2uint_rn(per_assign_wrong * (float) EDGE_WRONG_LEVEL);
	wrong = min(MAX_EDGE_WRONG, wrong);
	return set_bits(reg_stat, wrong, 0, 6);
}
__device__ __forceinline__ unsigned rs_set_processing(unsigned reg_stat, unsigned processing) {
	return set_bits(reg_stat, processing, 6, 11);
}
__device__ __forceinline__ unsigned rs_set_assigned(unsigned reg_stat, unsigned assigned) {
	return set_bits(reg_stat, assigned, 17, 14);
}
typedef struct wl_ct_t {
	unsigned* wl_data;
	unsigned* write_done;
	unsigned* read_done;
	unsigned* resv_ptr;
	unsigned* bk_ptr_array[NUM_BAG];
	unsigned long long* agm_buf;
	unsigned* regular_status;
	unsigned* done_work_count;
	unsigned* delta_info_broadcast;
	//total number of storage words
	unsigned wl_size;
	//total size of the liner space for each bag
	unsigned linear_size;
	unsigned num_block_linear;

	int num_tb_32;
	int num_tb_real;
	int num_warp;

	int nnode;
	int nedge;
	float ave_degree;
	float ave_wt;

	unsigned mini_grain;
	int delta_init;
	void alloc(int num_tb, int warp_per_tb, unsigned suggested_size, int nnode, int nedge);
	void reinit();
	void set_param(float ave_wt, float ave_degree);

	void init(int num_tb, int warp_per_tb, unsigned suggested_size, int delta, float ave_wt, float ave_degree, int nnode, int nedge);
	void free();

	__device__ void read_manager();
	__device__ void assign_manager(int start_node);
	__device__ void alloc_manager();
	__device__ void manager_init();
	__device__ void init_regular();

	//get work from wl
	__device__ unsigned get_assignment(CSRGraph& graph, unsigned long long &m_assignment, unsigned warp_id, unsigned* work_count = NULL, unsigned total_work = 0);
	__device__ unsigned pop_work(unsigned idx);
	__device__ void get_global(unsigned warp_id);

	//add work to wl
	__device__ void push_work(unsigned bag_id, unsigned node);
	__device__ unsigned translate_write_ptr(unsigned bag_id, unsigned m_ptr);
	__device__ void prefetch_bk_cache(unsigned bag_id, unsigned m_ptr);
	__device__ void update_bk_cache(unsigned bag_id, unsigned m_tag, unsigned cache_idx);
	__device__ void epilog(unsigned long long m_assignment, unsigned work_size, bool coop);
	//this is used by clean up only, once per BK
	__device__ unsigned get_bag_id(unsigned real_bk_idx);
	__device__ unsigned dist_to_bag_id_int(unsigned bag_id, node_data_type dst_dist);
	//tb coop
	__device__ void tb_coop_assign(int vertex_id, int first_edge, unsigned long long m_assignment, int& m_size, unsigned tb_coop_lane);
	__device__ void tb_coop_process(CSRGraph& graph, int warp_id);
} worklist;

__global__ void wl_kernel(worklist wl, int start_node) {
	wl.manager_init();
	if (thread_id_x() < (MAX_CON_BAG * 32)) {
		wl.read_manager();
	} else if (thread_id_x() < ((MAX_CON_BAG + 1) * 32)) {
		wl.assign_manager(start_node);
	} else if (thread_id_x() < ((MAX_CON_BAG + 2) * 32)) {
		wl.alloc_manager();
	}
}

void worklist::alloc(int num_tb, int warp_per_tb, unsigned suggested_size, int nnode, int nedge) {
	this->nnode = nnode;
	this->nedge = nedge;
	assert(num_tb < MAX_TB);
	assert(NUM_BAG == 32);
	//round up to next pow 2
	{
		unsigned size = r_up(suggested_size, BLOCK_SIZE);
		wl_size = 1;
		while (wl_size < size) {
			wl_size = wl_size << 1;
		}

		assert(wl_size < 1073741824);

		linear_size = wl_size;
		//num bk should be pow 2
		std::bitset < 32 > t(linear_size / BLOCK_SIZE);
		assert(t.count() == 1);
		assert(linear_size / BLOCK_SIZE >= 32);
		num_block_linear = linear_size / BLOCK_SIZE;

	}

	{
		std::bitset < 32 > t(linear_size);
		if (t.count() != 1) {
			printf("linear_size  %u, is not power of 2\n", linear_size);
			fflush(0);
			exit(-1);
		}
	}

	{
		std::bitset < 32 > t(SEG_SIZE);
		if (t.count() != 1) {
			printf("seg size %u, is not power of 2\n", SEG_SIZE);
			fflush(0);
			exit(-1);
		}
		if ((linear_size % SEG_SIZE) != 0) {
			printf("seg size %u is not multiple of linear_size %u\n", SEG_SIZE, linear_size);
			fflush(0);
			exit(-1);
		}

	}

	{
		std::bitset < 32 > t(BLOCK_SIZE);
		if (t.count() != 1) {
			printf("block size %u, is not power of 2\n", BLOCK_SIZE);
			fflush(0);
			exit(-1);
		}
		if ((linear_size % BLOCK_SIZE) != 0) {
			printf("block size %u is not multiple of linear_size %u\n",
			BLOCK_SIZE, linear_size);
			fflush(0);
			exit(-1);
		}

		if ((BLOCK_SIZE % (SEG_SIZE * 32)) != 0) {
			printf("block size %u is not multiple of seg *32 %u\n", BLOCK_SIZE, SEG_SIZE * 32);
			fflush(0);
			exit(-1);
		}
	}

	if (cudaMalloc((void **) &(delta_info_broadcast), sizeof(unsigned)) != cudaSuccess) {
		printf("alloc failure\n");
		fflush(0);
		exit(1);
	}

	num_tb_real = num_tb;
	num_tb_32 = r_up(num_tb_real, 32);
	num_warp = warp_per_tb;
	unsigned num_seg = wl_size / SEG_SIZE;
	unsigned num_block = wl_size / BLOCK_SIZE;

	if (cudaMalloc((void **) &(wl_data), wl_size * sizeof(unsigned)) != cudaSuccess) {
		printf("wl alloc failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMalloc((void **) &(write_done), num_seg * sizeof(unsigned)) != cudaSuccess) {
		printf("bitmask alloc failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMalloc((void **) &(resv_ptr), NUM_BAG * sizeof(unsigned)) != cudaSuccess) {
		printf("resv ptr alloc failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMalloc((void **) &(done_work_count), NUM_BAG * sizeof(unsigned)) != cudaSuccess) {
		printf("resv ptr alloc failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMalloc((void **) &(agm_buf), sizeof(unsigned long long) * num_tb_real) != cudaSuccess) {
		printf("wl alloc failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMemset(agm_buf, 0, sizeof(unsigned long long) * num_tb_real) != cudaSuccess) {
		printf("wl set 2  failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMalloc((void **) &(regular_status), sizeof(unsigned) * num_tb_32) != cudaSuccess) {
		printf("wl alloc failure\n");
		fflush(0);
		exit(1);
	}

	for (int bag = 0; bag < NUM_BAG; bag++) {
		if (cudaMalloc((void **) &(bk_ptr_array[bag]), num_block_linear * sizeof(unsigned)) != cudaSuccess) {
			printf("wl alloc failure\n");
			fflush(0);
			exit(1);
		}
	}
	if (cudaMalloc((void **) &(read_done), num_block * sizeof(uint)) != cudaSuccess) {
		printf("wl alloc failure\n");
		fflush(0);
		exit(1);
	}

}

void worklist::reinit() {

	unsigned num_seg = wl_size / SEG_SIZE;
	unsigned num_block = wl_size / BLOCK_SIZE;

	if (cudaMemset(write_done, 0, num_seg * sizeof(unsigned)) != cudaSuccess) {
		printf("bitmask set failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMemset(resv_ptr, 0, NUM_BAG * sizeof(unsigned)) != cudaSuccess) {
		printf("resv ptr set failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMemset(done_work_count, 0, NUM_BAG * sizeof(unsigned)) != cudaSuccess) {
		printf("resv ptr set failure\n");
		fflush(0);
		exit(1);
	}

	if (cudaMemset(agm_buf, 0, sizeof(unsigned long long) * num_tb_real) != cudaSuccess) {
		printf("wl set 2  failure\n");
		fflush(0);
		exit(1);
	}

	unsigned reg_stat_host[MAX_TB];
	for (int i = 0; i < num_tb_32; i++) {
		if (i < num_tb_real) {
			reg_stat_host[i] = 0x80000000;
		} else {
			reg_stat_host[i] = 0;
		}
	}
	cudaMemcpy(regular_status, reg_stat_host, sizeof(unsigned) * num_tb_32, cudaMemcpyHostToDevice);

	for (int bag = 0; bag < NUM_BAG; bag++) {
		if (cudaMemset(bk_ptr_array[bag], 0xff, num_block_linear * sizeof(unsigned)) != cudaSuccess) {
			printf("wl set 4 failure\n");
			fflush(0);
			exit(1);
		}
	}

	if (cudaMemset(read_done, 0, num_block * sizeof(unsigned)) != cudaSuccess) {
		printf("wl set 5 failure\n");
		fflush(0);
		exit(1);
	}

}
void worklist::set_param(float ave_wt, float ave_degree) {
	this->ave_wt = ave_wt;
	this->ave_degree = ave_degree;
	{
		float suggest_mini_grain = (float) 32 / ave_degree;
		int m_prev = 0;
		int m = 1;
		do {
			if (((float) m_prev <= suggest_mini_grain) && ((float) m > suggest_mini_grain)) {
				break;
			}
			m_prev = m;
			m <<= 1;
		} while (m <= 32);
		assert(m_prev <= 32);
		mini_grain = max(1, m_prev);

	}

	float f = (ave_wt / ave_degree) / INIT_FACTOR;
	unsigned pow2 = 1;
	unsigned u = 0;
	do {
		unsigned prev_pow2 = pow2 >> 1;
		if (((double) f >= (double) prev_pow2) && ((double) f <= (double) pow2)) {
			//within range
			if (((double) f - (double) prev_pow2) < ((double) pow2 - (double) f)) {
				u = prev_pow2;
			} else {
				u = pow2;
			}
			break;
		} else {
			pow2 = pow2 << 1;
		}
	} while (1);

	//must be larger than 32
	u = max(4, u);
	//u = delta;
	printf("float %f, round %u\n", f, u);
	{
		std::bitset < 32 > t(u);
		if (t.count() != 1) {
			printf("delta %u, is not power of 2\n", u);
			fflush(0);
			exit(-1);
		}
		delta_init = t._Find_first();
	}

	printf("multi-bag, multi-warp, variant c, wl size is %u, linear size is %u\n", wl_size, linear_size);
	printf("MAX_CON_BAG %u, MINI_GRAIN %u, ave_wt %f, ave_degree %f, delta init %u (%f)\n", MAX_CON_BAG, mini_grain, ave_wt, ave_degree, delta_init, f);

}

void worklist::free() {
	cudaFree(wl_data);
	cudaFree(write_done);
	cudaFree(resv_ptr);
	cudaFree(agm_buf);
	cudaFree(read_done);
	cudaFree(done_work_count);
	cudaFree(delta_info_broadcast);

}

#define MB_CACHE_SIZE 16
__shared__ unsigned manager_exit;
__shared__ unsigned cur_bag_mtb;
__shared__ unsigned read_ptr[NUM_BAG];
__shared__ unsigned assign_ptr[NUM_BAG];
__shared__ unsigned manager_bk_cache[NUM_BAG][MB_CACHE_SIZE];
__shared__ unsigned shift_delta_local;
__shared__ unsigned start_signal;
__device__ void worklist::manager_init() {
	if (thread_id_x() < 32) {
		for (int bag = get_lane_id(); bag < NUM_BAG; bag += 32) {
			read_ptr[bag] = 0;
			assign_ptr[bag] = 0;
			for (int i = 0; i < MB_CACHE_SIZE; i++) {
				manager_bk_cache[bag][i] = NOT_FOUND;
			}
		}
		if (get_lane_id() == 0) {
			manager_exit = 0;
			cur_bag_mtb = 0;
			start_signal = 0;
		}
	}
	__syncthreads();
}

__device__ void worklist::alloc_manager() {
	//init
	__shared__ unsigned free_bits[BIT_CACHE_SIZE];

	unsigned num_block = wl_size / BLOCK_SIZE;
	unsigned num_block_word = round_up(num_block, 32 * 32) / 32;
	//the current allocated block boundary
	//unsigned alloc_bk = 0;
	//unsigned read_done_bk = 0;
	//int num_alloc_bk = 0;

	__shared__ unsigned alloc_bk[NUM_BAG];
	__shared__ unsigned read_done_bk[NUM_BAG];

	for (int bag = get_lane_id(); bag < NUM_BAG; bag += 32) {
		alloc_bk[bag] = 0;
		read_done_bk[bag] = 0;
	}
	int num_alloc_bk = 0;

	//the bit check ptr
	unsigned last_word_check = 0;
	for (int word = get_lane_id(); word < num_block_word; word += 32) {
		if (((word + 1) * 32) <= num_block) {
			//all 1
			free_bits[word] = FULL_MASK;
		} else if ((word * 32) >= num_block) {
			//all 0
			free_bits[word] = 0;
		} else {
			//part 1
			unsigned mask = 0;
			for (int bit = 0; bit < 32; bit++) {
				int block_idx = word * 32 + bit;
				if (block_idx < num_block) {
					mask = set_bits(mask, FULL_MASK, bit, 1);
				}
			}
			free_bits[word] = mask;
		}
	}

	__syncwarp();
	while (manager_exit == 0) {
		//alloc
		//check all bags
		unsigned m_bag = get_lane_id();
		bool need_alloc = false;
		unsigned resv_bk = (cub::ThreadLoad<cub::LOAD_CG>(&(resv_ptr[m_bag]))) / BLOCK_SIZE;
		unsigned pre_alloc_bk = resv_bk + PRE_ALLOC_DIST;
		pre_alloc_bk = pre_alloc_bk & (num_block_linear - 1);
		//alloc bk always lag behid pre_alloc_bk
		//pre-alloc blocks
		//find a free block
		need_alloc = (pre_alloc_bk != alloc_bk[m_bag]);

		unsigned alloc_mask = __ballot_sync(FULL_MASK, need_alloc);
		//alloc for each request
		for (unsigned pos = 1; find_nth_bit(alloc_mask, 0, pos) != NOT_FOUND; pos++) {
			unsigned bag_id = find_nth_bit(alloc_mask, 0, pos);
			//num bk word always round to 32
			for (int i = get_lane_id(); i < num_block_word; i += 32) {
				int word_idx = i + last_word_check;
				if (word_idx >= num_block_word) {
					word_idx -= num_block_word;
				}
				//start from last checked word
				unsigned bitmask = free_bits[word_idx];
				//find a 1
				unsigned bit_idx = find_nth_bit(bitmask, 0, 1);
				unsigned vote_result = __ballot_sync(FULL_MASK, (bit_idx != NOT_FOUND));
				unsigned lane_idx = find_nth_bit(vote_result, 0, 1);
				if (lane_idx != NOT_FOUND) {
					//find a free block
					unsigned block_idx = ((word_idx * 32) + bit_idx);
					if (lane_idx == get_lane_id()) {
						//allocated bk addr
						unsigned ptr = block_idx * BLOCK_SIZE;
						unsigned alloc_idx = alloc_bk[bag_id];
						cub::ThreadStore<cub::STORE_CG>(&(bk_ptr_array[bag_id][alloc_idx]), ptr);
						//set the bit
						free_bits[word_idx] = set_bits(bitmask, 0, bit_idx, 1);
						//update the alloc bk ptr
						alloc_bk[bag_id] = (alloc_idx + 1) & (num_block_linear - 1);
					}
					__syncwarp();
					last_word_check = (word_idx & ~(32 - 1));
					num_alloc_bk++;
					break;
				}
			}
		}
		{
			//free
			__syncwarp();
			unsigned done_bk = read_done_bk[m_bag];
			unsigned real_bk_idx = cub::ThreadLoad<cub::LOAD_CG>(&(bk_ptr_array[m_bag][done_bk])) / BLOCK_SIZE;
			unsigned done_count = cub::ThreadLoad<cub::LOAD_CG>(&(read_done[real_bk_idx]));
			bool done = (done_count == BLOCK_SIZE);
			//handle all read done
			unsigned done_mask = __ballot_sync(FULL_MASK, done);
			num_alloc_bk -= count_bit(done_mask);
			while (done_mask != 0) {
				unsigned target_bag = find_ms_bit(done_mask);
				unsigned target_real_bk_idx = __shfl_sync(FULL_MASK, real_bk_idx, target_bag);
				//reset write_done counters
				unsigned seg_start = (target_real_bk_idx * BLOCK_SIZE) / SEG_SIZE;
				for (int seg_idx = get_lane_id(); seg_idx < (BLOCK_SIZE / SEG_SIZE); seg_idx += 32) {
					unsigned seg_word = seg_start + seg_idx;
					cub::ThreadStore<cub::STORE_CG>(&(write_done[seg_word]), (unsigned) 0);
				}
				done_mask = set_bits(done_mask, 0, target_bag, 1);
			}

			if (done) {
				unsigned word_idx = real_bk_idx / 32;
				unsigned bit_idx = real_bk_idx % 32;
				unsigned mask = set_bits(0, FULL_MASK, bit_idx, 1);
				atomicOr(&(free_bits[word_idx]), mask);
				cub::ThreadStore<cub::STORE_CG>(&(read_done[real_bk_idx]), (unsigned) 0);
				cub::ThreadStore<cub::STORE_CG>(&(bk_ptr_array[m_bag][done_bk]), (unsigned) NOT_FOUND);
				read_done_bk[m_bag] = (done_bk + 1) & (num_block_linear - 1);
			}


		}
		if (((num_block - num_alloc_bk)) < (NUM_BAG)) {
			if (get_lane_id() == 0) {
				printf("running out memory, try increase WL_SIZE_MUL in kernel.cu\n");
			}
			assert(false);
		}

	}
}

//cached block ptr
__device__ __forceinline__ unsigned bc_get_tag(unsigned entry) {
	return extract_bits(entry, 0, 16);
}

__device__ __forceinline__ unsigned bc_get_idx(unsigned entry) {
	return extract_bits(entry, 16, 16);
}

__device__ __forceinline__ unsigned bc_set_tag(unsigned entry, unsigned tag) {
	return set_bits(entry, tag, 0, 16);
}

__device__ __forceinline__ unsigned bc_set_idx(unsigned entry, unsigned idx) {
	return set_bits(entry, idx, 16, 16);
}

__device__ void worklist::read_manager() {

#if PRINT_DEBUG  == 1
	int print_counter = 0;
#endif

#if PRINT_BAG == 1
	unsigned bag_total_work = 0;
#endif

	//wait for start
	volatile unsigned can_start = start_signal;
	do {

		can_start = start_signal;
		__threadfence_block();
	} while (can_start == 0);

	//each warp is guaranteed to work on a different bag
	unsigned m_bag = thread_id_x() / 32;
	unsigned total_threads = num_tb_real * num_warp * 32;
	while (manager_exit == 0) {
		__syncwarp();
		unsigned m_read_ptr = read_ptr[m_bag];
		if ((m_read_ptr - assign_ptr[m_bag]) < (total_threads * 2)) {
			unsigned m_tag = m_read_ptr / BLOCK_SIZE;
			unsigned cache_idx = m_tag % MB_CACHE_SIZE;
			unsigned entry = manager_bk_cache[m_bag][cache_idx];
			unsigned cur_tag = bc_get_tag(entry);
			if (cur_tag == m_tag) {
				unsigned translated_upper_ptr = bc_get_idx(entry) * BLOCK_SIZE;
				unsigned cur_done_work = NOT_FOUND;
				unsigned cur_resv_ptr = NOT_FOUND;
				unsigned read_upper_ptr = m_read_ptr & ~(BLOCK_SIZE - 1);
				unsigned read_lower_ptr = m_read_ptr % BLOCK_SIZE;
				unsigned lower_line_ptr = read_lower_ptr & ~(SEG_SIZE * 32 - 1);
				unsigned m_word = (translated_upper_ptr + lower_line_ptr) / SEG_SIZE + get_lane_id();
				unsigned word_offset = 0;
				unsigned sub_count = 0;
				for (; lower_line_ptr < BLOCK_SIZE; lower_line_ptr += (32 * SEG_SIZE)) {
					unsigned count = cub::ThreadLoad<cub::LOAD_CG>(&(write_done[m_word]));
					unsigned vote_result = __ballot_sync(FULL_MASK, (count != SEG_SIZE));
					if (vote_result != 0) {
						word_offset = find_nth_bit(vote_result, 0, 1);
						sub_count = __shfl_sync(FULL_MASK, count, word_offset);
						break;
					}
					m_word += 32;
				}
				unsigned new_lower_ptr = lower_line_ptr + (word_offset * SEG_SIZE);
				if (new_lower_ptr <= read_lower_ptr) {
					//proposed new read ptr
					new_lower_ptr += sub_count;
					//check the sub line
					//need to ensure mem ordering
					__threadfence();
					cur_resv_ptr = cub::ThreadLoad<cub::LOAD_CG>(&(resv_ptr[m_bag]));
					cur_done_work = cub::ThreadLoad<cub::LOAD_CG>(&(done_work_count[m_bag]));
					//if not match, then not safe to update
					if (cur_resv_ptr != (read_upper_ptr + new_lower_ptr)) {
						new_lower_ptr = read_lower_ptr;
					}
				}

				if (new_lower_ptr != read_lower_ptr) {
					//read ptr changes, update
					if (get_lane_id() == 0) {
						read_ptr[m_bag] = read_upper_ptr + new_lower_ptr;
					}

				} else {
					//see if we need to update bag
					//we update only if we are the head
					if (m_bag == (cur_bag_mtb % NUM_BAG)) {
						if ((assign_ptr[m_bag] == cur_resv_ptr) && (cur_done_work == assign_ptr[m_bag])) {
							if (get_lane_id() == 0) {
								//update cur_bag
								cur_bag_mtb++;
							}
							//now i will check after the batch
							m_bag = (m_bag + MAX_CON_BAG) % NUM_BAG;

						}
					}
				}
			} else {
				//refill from assign ptr
				if (get_lane_id() < MB_CACHE_SIZE) {
					unsigned m_tag = (assign_ptr[m_bag] + (get_lane_id() * BLOCK_SIZE)) / BLOCK_SIZE;
					unsigned cache_idx = m_tag % MB_CACHE_SIZE;
					unsigned bk_array_idx = m_tag & (num_block_linear - 1);	//Virtual bk idx
					unsigned real_ptr = cub::ThreadLoad<cub::LOAD_CG>(&(bk_ptr_array[m_bag][bk_array_idx]));
					if (real_ptr != NOT_FOUND) {
						//now we can update the cache
						unsigned real_idx = real_ptr / BLOCK_SIZE;
						unsigned entry = 0;
						entry = bc_set_idx(entry, real_idx);
						entry = bc_set_tag(entry, m_tag);
						//write to cache
						manager_bk_cache[m_bag][cache_idx] = entry;
					}
				}
				__syncwarp();
			}
		}

#if PRINT_DEBUG  ==  1
		if (print_counter == 0) {
			if (get_lane_id() == 0) {
				unsigned m_tag = read_ptr[m_bag] / BLOCK_SIZE;
				unsigned cache_idx = m_tag % MB_CACHE_SIZE;
				unsigned entry = manager_bk_cache[m_bag][cache_idx];
				unsigned cur_tag = bc_get_tag(entry);
				unsigned cur_idx = bc_get_idx(entry);
				unsigned virtual_idx = m_tag & (num_block_linear - 1);

				printf(
						"cur bag %u, m_bag %u, read ptr %u, assign_ptr %u, reserve ptr %u, done work %u, read ptr m_tag %u, cur_tag %u, virtual idx %u, real idx %u\n",
						cur_bag_mtb, m_bag, read_ptr[m_bag], assign_ptr[m_bag],
						cub::ThreadLoad<cub::LOAD_CG>(&(resv_ptr[m_bag])),
						cub::ThreadLoad<cub::LOAD_CG>(
								&(done_work_count[m_bag])), m_tag, cur_tag,
						virtual_idx, cur_idx);
			}
			__syncwarp();
		}
		print_counter++;
		//if (print_counter == ) {
		if (print_counter == 1024) {
			print_counter = 0;
		}
#endif
	}
	asm volatile("exit;");
}

#define MAX_SHIFT 31
#define MIN_SHIFT 0
__device__ __forceinline__ int inc_delta_shift(int shift, int amount) {
	return min(MAX_SHIFT, shift + amount);
}
__device__ __forceinline__ int dec_delta_shift(int shift, int amount) {
	return max(MIN_SHIFT, shift - amount);
}

#define PROFILE_WORK 10
#define CHANGE_BAG_WAIT 1
#define CHANGE_DELTA_WAIT 1
#define PROBE_LOW_WORK_MIN 8
#define CONSEC_LOW_BAG_TSH 4
#define AGG_LOW_BAG_LEAVE 1
#define CONSEC_LOW_PROB_TSH (CONSEC_LOW_BAG_TSH*AGG_LOW_BAG_LEAVE + 5)

#define HIGH_CLIP 0.7f
#define HIGH_CLIP_RANGE 6
#define LOW_DELTA_BAN 4

#define INIT_SKIP_CONBAG MAX_CON_BAG
#define MIN_CON_BAG 2

#define INIT_MUL 0.5f

#define UTL_LINEAR_TSH 16.0f
#define LINEAR_LOW_P_COEFF 0.03f
#define LINEAR_LOW_UTL_COEFF 0.1f
#define LINEAR_HI_P_COEFF 0.08f
#define LINEAR_HI_UTL_COEFF 0.4f

#define CURVE_HI_P_COEFF 0.4f
#define CURVE_HI_UTL_COEFF 1.75f
#define CURVE_HI_P_POW 0.6f
#define CURVE_HI_UTL_POW 0.6f

#define CURVE_LOW_P_COEFF 0.1f
#define CURVE_LOW_UTL_COEFF 0.4f
#define CURVE_LOW_P_POW 0.1f
#define CURVE_LOW_UTL_POW 0.1f

#define STATE_RESET 0
#define STATE_PRE_DEC_DELTA_1 1
#define STATE_PRE_DEC_DELTA_2 2
#define STATE_DEC_DELTA 3

#define INC_DELTA_MIN_ITER 6
#define CONFIG_AVE (1.0f / INC_DELTA_MIN_ITER)
#define INC_DELTA_WAIT_BAG 3
#define INC_DELTA_BAN_BAG 6
#define DEC_DELTA_DECLINE 0.7f
#define DEC_DELTA_REMAIN_MUL 1.5f
#define DEC_DELTA_FORGIVE 1

#define SMALL_DELTA_MUL 1.0f
#define SMALL_DELTA_INC_TSH (INIT_SKIP_CONBAG+3)
#define SMALL_WAIT_BAG_FROM_PROBE 58
#define SMALL_WAIT_WORK_FROM_PROBE 4.0f

#define CLEAR_LOW_INTERVAL 0.1f
#define SAME_BAG_DEC_INTERVAL 0.6f

#define INIT_MEASURE_TB 1.0f
#define SM_MEASURE_COFF 1000000.0f
//#define SM_MEASURE_COFF 4000000.0f
//#define SM_MEASURE_COFF 1000000.0f
#define MAX_SM_TB 6.0f
#define MIN_SM_TB 1.0f
#define FIXED_DELTA_EDGE_MUL 50.0f
#define FIXED_DELTA_NODE_MUL 4.0f
#define FIXED_DELTA_EDGE_MUL2 100.0f
#define FIXED_DELTA_NODE_MUL2 6.0f

__device__ void worklist::assign_manager(int start_node) {
	__syncwarp();
	__shared__ unsigned tb_list[MAX_TB];
	__shared__ unsigned new_work[MAX_TB];
	__shared__ unsigned tb_status[MAX_TB];
	__shared__ unsigned local_seq[MAX_TB];
	for (int tb = get_lane_id(); tb < num_tb_32; tb += 32) {
		new_work[tb] = 0;
		local_seq[tb] = 0;
		tb_status[tb] = 0;
	}
	unsigned last_assigned_tb = 0;

	__shared__ unsigned bag_dist[NUM_BAG];
	bag_dist[get_lane_id()] = 0;

	bool delta_change = false;
	int cur_delta = delta_init;
	int next_delta = delta_init;

	//unsigned measure_tb = (unsigned) ((float) num_tb_real * 1.0f);

	////////////////////
	__shared__ unsigned low_delta_violations[32];
	__shared__ unsigned last_assign_ptr[32];
	low_delta_violations[get_lane_id()] = 0;
	last_assign_ptr[get_lane_id()] = 0;
	//must be init to non_zero to avoid nan
	typedef cub::WarpReduce<unsigned> WarpReduce;
	__shared__ typename WarpReduce::TempStorage temp_storage;

	//////////////////////////
	unsigned measure_tb = 1;
	unsigned old_resv_ptr = 0;
	unsigned cur_dist = 0;
	int current_bag = 0;
	unsigned total_assigned_work = 0;
	unsigned concur_bag = 1;
	unsigned prof_iter = 2;
	unsigned prof_assigned_tb = 0;
	unsigned prof_assigned_work = 0;
	unsigned last_change_delta_iter = 0;
	unsigned last_change_bag_iter = prof_iter;

	bool probe_low_phase = true;
	unsigned probe_low_work = (unsigned) ((float) PROFILE_WORK / ave_degree) * 3;
	probe_low_work = max(PROBE_LOW_WORK_MIN, probe_low_work);
	unsigned consec_low_bag = 0;
	unsigned agg_low_bag = 0;
	unsigned consec_low_prob = 0;

	float util_divident = (float) num_tb_real * num_warp * 32 / ave_degree;
	float low_P;
	float low_UTL;
	float hi_P;
	float hi_UTL;
	if (ave_degree >= UTL_LINEAR_TSH) {
		low_P = ave_degree * LINEAR_LOW_P_COEFF;
		low_UTL = ave_degree * LINEAR_LOW_UTL_COEFF;
		hi_P = ave_degree * LINEAR_HI_P_COEFF;
		hi_UTL = ave_degree * LINEAR_HI_UTL_COEFF;
	} else {
		low_P = CURVE_LOW_P_COEFF * pow(ave_degree, CURVE_LOW_P_POW);
		low_UTL = CURVE_LOW_UTL_COEFF * pow(ave_degree, CURVE_LOW_UTL_POW);
		hi_P = CURVE_HI_P_COEFF * pow(ave_degree, CURVE_HI_P_POW);
		hi_UTL = CURVE_HI_UTL_COEFF * pow(ave_degree, CURVE_HI_UTL_POW);
	}
	unsigned cur_state = STATE_RESET;
	int last_inc_delta_bag = -10;
	int last_dec_delta_bag = -10;
	unsigned remaining_work;
	unsigned in_range_assign;
	bool invite_check = false;
	bool start_state_machine = false;
	float last_config_util;
	float transit_util;
	float forgive_count;
	//push start node

	bool fixed_delta = false;
	if ((nnode < ((num_tb_real * num_warp * 32) * FIXED_DELTA_NODE_MUL)) || (nedge < ((num_tb_real * num_warp * 32) * FIXED_DELTA_EDGE_MUL))) {
		fixed_delta = true;
	} else if ((nnode < ((num_tb_real * num_warp * 32) * FIXED_DELTA_NODE_MUL2)) && (nedge < ((num_tb_real * num_warp * 32) * FIXED_DELTA_EDGE_MUL2))) {
		fixed_delta = true;
	}

	if (fixed_delta) {
		cur_delta = inc_delta_shift(delta_init, FIXED_SHIFT);
		next_delta = inc_delta_shift(delta_init, FIXED_SHIFT);
		concur_bag = MAX_CON_BAG;
		probe_low_phase = false;
	}

	if (get_lane_id() == 0) {
		atomicAdd(&(resv_ptr[0]), 1);
		cub::ThreadStore<cub::STORE_CG>(&(wl_data[0]), (unsigned) start_node);
		atomicAdd(&(write_done[0]), 1);
		__threadfence_block();
		start_signal = 1;
		unsigned delta_info = 0;
		delta_info = bfg_set_delta(delta_info, cur_delta);
		cub::ThreadStore<cub::STORE_CG>(delta_info_broadcast, delta_info);
	}
	__syncwarp();

	unsigned config_agg_prof_iter = 0;
	float config_agg_P = 0.0f;
	float config_agg_A = 0.0f;

	unsigned small_delta_counter = 0;
	unsigned small_start_bag;
	unsigned small_start_work;
	unsigned small_delta_work = (unsigned) (SMALL_DELTA_MUL * (float) (num_tb_real * num_warp * 32));

	unsigned next_dec_delta_work = 0;
	unsigned next_clear_vio_work = ((float) nnode * INIT_MUL);

	while (manager_exit == 0) {
		__syncwarp();
		int updated_bag = cur_bag_mtb;
		float cur_A = 0;
		float cur_P = 0;

		if (last_assign_ptr[updated_bag % NUM_BAG] < read_ptr[updated_bag % NUM_BAG]) {
			if (!fixed_delta) {
				if ((prof_assigned_tb >= measure_tb) || ((current_bag != updated_bag))) {

					unsigned m_bag = (current_bag + get_lane_id()) % NUM_BAG;
					unsigned new_resv_ptr = cub::ThreadLoad<cub::LOAD_CG>(&(resv_ptr[m_bag]));
					unsigned produced_work = new_resv_ptr - old_resv_ptr;
					unsigned warp_produced_work = WarpReduce(temp_storage).Sum(produced_work);
					warp_produced_work = __shfl_sync(FULL_MASK, warp_produced_work, 0);
					unsigned cur_assign_ptr = assign_ptr[m_bag];
					unsigned all_work = new_resv_ptr - cur_assign_ptr;
					total_assigned_work += prof_assigned_work;
					//update bag dist
					{
						if (warp_produced_work > 0) {
							unsigned m_dist = cur_dist + get_lane_id() * (1 << cur_delta);

							//set delta
							//see if dist equals
							if ((produced_work > 0) && (get_lane_id() != 0)) {
								bag_dist[m_bag] = m_dist;
							}
							cur_delta = next_delta;
						}
					}

					unsigned A_count = 0;
					unsigned P_count = 0;
					for (int tb = get_lane_id(); tb < num_tb_real; tb += 32) {
						unsigned m_status = tb_status[tb];
						A_count += rs_get_assigned(m_status);
						P_count += rs_get_processing(m_status);
						A_count += new_work[tb];
					}

					unsigned A_sum = WarpReduce(temp_storage).Sum(A_count);
					unsigned P_sum = WarpReduce(temp_storage).Sum(P_count);
					A_sum = __shfl_sync(FULL_MASK, A_sum, 0);
					P_sum = __shfl_sync(FULL_MASK, P_sum, 0);
					cur_A = (float) A_sum / util_divident;
					cur_P = (float) P_sum / util_divident;

					if (config_agg_prof_iter == 0) {
						config_agg_A = cur_A;
						config_agg_P = cur_P;
					} else {
						config_agg_A = (1.0f - CONFIG_AVE) * config_agg_A + CONFIG_AVE * cur_A;
						config_agg_P = (1.0f - CONFIG_AVE) * config_agg_P + CONFIG_AVE * cur_P;
					}
					config_agg_prof_iter++;

					if (!start_state_machine) {
						unsigned init_assign_tsh = (unsigned) ((float) nnode * INIT_MUL);
						if (total_assigned_work > init_assign_tsh) {
							start_state_machine = true;
							float coeff = (float) nedge / SM_MEASURE_COFF;
							coeff = min(MAX_SM_TB, coeff);
							coeff = max(MIN_SM_TB, coeff);
							measure_tb = num_tb_real * __float2uint_rn(coeff);
							config_agg_prof_iter = 0;
						}
					}

					//we aggregated enough work, so we can profile for delta
					//use the previous slot

					if (((prof_iter - last_change_delta_iter) > CHANGE_DELTA_WAIT) && (cur_delta == next_delta)) {

						if (probe_low_phase) {
							if ((updated_bag - current_bag) >= (NUM_BAG - 1)) {
								consec_low_bag = 0;
								consec_low_prob = 0;
								next_delta = inc_delta_shift(cur_delta, 2);
								delta_change = true;
								last_change_delta_iter = prof_iter;
								agg_low_bag++;
							} else if (warp_produced_work < PROFILE_WORK) {
								if (updated_bag != current_bag) {
									consec_low_bag++;
								}
								consec_low_prob++;
								probe_low_work++;
								if (consec_low_prob >= CONSEC_LOW_PROB_TSH) {
									probe_low_phase = false;
									concur_bag = INIT_SKIP_CONBAG;
								} else if (consec_low_bag >= CONSEC_LOW_BAG_TSH) {
									next_delta = inc_delta_shift(cur_delta, 1);
									delta_change = true;
									last_change_delta_iter = prof_iter;
									consec_low_bag = 0;
									agg_low_bag++;
									if (agg_low_bag >= AGG_LOW_BAG_LEAVE) {
										probe_low_phase = false;
										concur_bag = INIT_SKIP_CONBAG;
									}
								}

							} else {
								consec_low_prob = 0;
								consec_low_bag = 0;
								//find lower delta first
								float m_percent = (float) produced_work / (float) warp_produced_work;
								unsigned high_clip_mask = __ballot_sync(FULL_MASK, m_percent > HIGH_CLIP);
								unsigned violation_lane = find_ms_bit(high_clip_mask);
								if ((violation_lane != NOT_FOUND) && (violation_lane > HIGH_CLIP_RANGE)) {
									//we have a violation
									if (get_lane_id() == 0) {
										low_delta_violations[cur_delta] += 1;
									}
									if (cur_delta == MAX_SHIFT) {
										//hit top
										probe_low_phase = false;
										concur_bag = MIN_CON_BAG;
										//agg_hi_count++;
									} else {
										agg_low_bag++;
										next_delta++;
										delta_change = true;
										last_change_delta_iter = prof_iter;
									}
								} else {
									//try to decrease, until find the first ban
									if ((cur_delta == MIN_SHIFT) || (low_delta_violations[cur_delta - 1] >= LOW_DELTA_BAN)) {
										//we have a ban or hit bottom, no change, and get out
										probe_low_phase = false;
										concur_bag = MIN_CON_BAG;
									} else {
										next_delta--;
										delta_change = true;
										last_change_delta_iter = prof_iter;
									}
								}
							}

							if (!probe_low_phase) {
								small_start_bag = current_bag + SMALL_WAIT_BAG_FROM_PROBE;
								small_start_work = total_assigned_work + (unsigned) (SMALL_WAIT_WORK_FROM_PROBE * (float) (num_tb_real * num_warp * 32));
								measure_tb = (unsigned) ((float) num_tb_real * INIT_MEASURE_TB);
							}
						} else {

							//find lower delta first
							//generative distribution analysis

							bool low_delta = false;
							if (invite_check && ((prof_iter - last_change_bag_iter) > CHANGE_BAG_WAIT) && (warp_produced_work > PROFILE_WORK)) {
								invite_check = false;
								//find high clip violation
								float m_percent = (float) produced_work / (float) warp_produced_work;
								unsigned high_clip_mask = __ballot_sync(FULL_MASK, m_percent > HIGH_CLIP);
								unsigned violation_lane = find_ms_bit(high_clip_mask);
								if ((violation_lane != NOT_FOUND) && (violation_lane > HIGH_CLIP_RANGE)) {
									//we have a violation
									low_delta = true;
								}
								//handle low delta
								if (low_delta) {
									if (get_lane_id() == 0) {
										low_delta_violations[cur_delta] += 1;
									}
									if (cur_delta < MAX_SHIFT) {
										next_delta++;
										delta_change = true;
										small_delta_counter = 0;
										config_agg_prof_iter = 0;
										last_change_delta_iter = prof_iter;
										if (start_state_machine) {
											next_clear_vio_work = total_assigned_work + (unsigned) ( CLEAR_LOW_INTERVAL * 2 * (float) nnode);
										}
										last_inc_delta_bag = updated_bag;
									}
									cur_state = STATE_RESET;
								}

							}

							if ((updated_bag != current_bag) && (small_delta_counter >= SMALL_DELTA_INC_TSH)) {

								//inc detla
								if (cur_delta < MAX_SHIFT) {
									next_delta++;
									delta_change = true;
									config_agg_prof_iter = 0;
									last_inc_delta_bag = updated_bag;
									cur_state = STATE_RESET;
									small_delta_counter = 0;
									last_change_delta_iter = prof_iter;
								}

							}

							//clear violation
							if (total_assigned_work > next_clear_vio_work) {
								if (get_lane_id() == 0) {
									if ((cur_delta > MIN_SHIFT) && (low_delta_violations[cur_delta - 1] >= LOW_DELTA_BAN)) {
										low_delta_violations[cur_delta - 1] = LOW_DELTA_BAN - 1;
									}
								}
								__syncwarp();
								next_clear_vio_work = total_assigned_work + (unsigned) ( CLEAR_LOW_INTERVAL * (float) nnode);
							}

							if (start_state_machine && ((prof_iter - last_change_delta_iter) > CHANGE_DELTA_WAIT)) {
								//utilization analysis
								switch (cur_state) {
								case STATE_RESET: {
									float util = cur_P + cur_A;
									if ((cur_P < low_P) && (util < low_UTL)) {
										if (concur_bag < MAX_CON_BAG) {
											concur_bag++;
											config_agg_prof_iter = 0;
											last_change_bag_iter = prof_iter;
										} else if ((updated_bag != current_bag) && (cur_delta < MAX_SHIFT) && (config_agg_prof_iter >= INC_DELTA_MIN_ITER)) {
											//change delta check
											//must be different
											float config_UTL = config_agg_A + config_agg_P;
											if ((config_agg_P < low_P) && (config_UTL < low_UTL) && (((int) updated_bag - last_inc_delta_bag) >= INC_DELTA_WAIT_BAG)
													&& (((int) updated_bag - last_dec_delta_bag) >= INC_DELTA_BAN_BAG)) {
												next_delta++;
												delta_change = true;
												config_agg_prof_iter = 0;
												small_delta_counter = 0;
												last_inc_delta_bag = updated_bag;
												last_change_delta_iter = prof_iter;
												next_clear_vio_work = total_assigned_work + (unsigned) ( CLEAR_LOW_INTERVAL * 2 * (float) nnode);
											}

										}
									} else if ((cur_P > hi_P) || (util > hi_UTL)) {
										if (concur_bag > MIN_CON_BAG) {
											concur_bag--;
											config_agg_prof_iter = 0;
											last_change_bag_iter = prof_iter;
										} else if ((cur_delta > MIN_SHIFT) && (low_delta_violations[cur_delta - 1] < LOW_DELTA_BAN)
												&& (((int) current_bag > last_dec_delta_bag) || (total_assigned_work > next_dec_delta_work))) {
											last_config_util = util;
											transit_util = util;
											forgive_count = 0;
											cur_state = STATE_PRE_DEC_DELTA_1;
										}
									}
								}
									break;

								case STATE_PRE_DEC_DELTA_1: {
									float util = cur_P + cur_A;
									if ((cur_P < hi_P) && (util < hi_UTL)) {
										//return back
										cur_state = STATE_RESET;
									} else {
										//wait increasing
										if (util > transit_util) {
											transit_util = util;
										} else {
											forgive_count++;
											if (forgive_count > DEC_DELTA_FORGIVE) {
												//go to next phase to wait for decreasing
												forgive_count = 0;
												transit_util = util;
												cur_state = STATE_PRE_DEC_DELTA_2;
											}
										}
									}
								}
									break;

								case STATE_PRE_DEC_DELTA_2: {
									float util = cur_P + cur_A;
									if ((cur_P < hi_P) && (util < hi_UTL)) {
										//return back
										cur_state = STATE_RESET;
									} else {
										//wait decreasing
										if (util < transit_util) {
											transit_util = util;
										} else {
											forgive_count++;
											if (forgive_count > DEC_DELTA_FORGIVE) {
												//dec delta for sure
												next_delta--;
												delta_change = true;
												last_dec_delta_bag = updated_bag;
												next_dec_delta_work = total_assigned_work + (unsigned) (SAME_BAG_DEC_INTERVAL * (float) nnode);
												small_delta_counter = 0;
												last_change_delta_iter = prof_iter;
												last_config_util = util;
												config_agg_prof_iter = 0;
												invite_check = true;
												/////
												unsigned cur_in_range_work = all_work;
												if (get_lane_id() >= concur_bag) {
													cur_in_range_work = 0;
												}
												remaining_work = WarpReduce(temp_storage).Sum(cur_in_range_work);
												remaining_work = __shfl_sync(
												FULL_MASK, remaining_work, 0);
												remaining_work = (unsigned) ((float) remaining_work * DEC_DELTA_REMAIN_MUL);
												in_range_assign = 0;
												////
												cur_state = STATE_DEC_DELTA;
											}
										}

									}
								}
									break;

								case STATE_DEC_DELTA: {
									in_range_assign += prof_assigned_work;
									float util = cur_P + cur_A;
									if ((util < (last_config_util * DEC_DELTA_DECLINE)) || (in_range_assign > remaining_work)) {
										//return back
										cur_state = STATE_RESET;
									}

								}
									break;

								default:
									assert(false);
								}
							}
						}
					}

					__syncwarp();
					//profile epilog
					//new resv ptr became old resv ptr, shuffle according to
					//potential bag change
					old_resv_ptr = __shfl_sync(FULL_MASK, new_resv_ptr, (get_lane_id() + (updated_bag - current_bag)) % 32);
					prof_assigned_tb = 0;
					prof_assigned_work = 0;
					prof_iter++;

				}

			}

			__syncwarp();
			if ((!probe_low_phase) && (updated_bag != current_bag) && ((prof_iter - last_change_delta_iter) > CHANGE_DELTA_WAIT)
					&& ((current_bag > small_start_bag) || (total_assigned_work > small_start_work))) {
				unsigned bag_assigned_work = assign_ptr[current_bag % NUM_BAG] - last_assign_ptr[current_bag % NUM_BAG];
				if (bag_assigned_work < small_delta_work) {
					small_delta_counter++;
				} else {
					small_delta_counter = 0;
				}

			}

			if ((updated_bag != current_bag) || delta_change) {

				if (updated_bag != current_bag) {
					unsigned bag_id = updated_bag % NUM_BAG;
					unsigned bag_offset = updated_bag - current_bag;
					//prevent jump
					if (((updated_bag - current_bag) > 25) && ((float) (1 << cur_delta)) > ave_wt) {
						bag_offset = 0;
					}
					unsigned new_offset_dist = cur_dist + (bag_offset << cur_delta);
					if ((new_offset_dist < bag_dist[bag_id]) || (bag_dist[bag_id] < cur_dist) || fixed_delta) {
						cur_dist = new_offset_dist;
					} else {
						cur_dist = bag_dist[bag_id];
					}
					if (get_lane_id() == 0) {
						last_assign_ptr[current_bag % NUM_BAG] = assign_ptr[current_bag % NUM_BAG];
					}
				}
				unsigned cur_dist_shift = (unsigned) (cur_dist >> next_delta);
				while (extract_bits(cur_dist_shift, 21, 32) != 0) {
					next_delta++;
					cur_delta = next_delta;
					cur_dist_shift = (unsigned) (cur_dist >> next_delta);
				}

				unsigned delta_info = cur_dist_shift;
				delta_info = bfg_set_bag(delta_info, updated_bag % NUM_BAG);
				delta_info = bfg_set_delta(delta_info, next_delta);

				if (get_lane_id() == 0) {
					cub::ThreadStore<cub::STORE_CG>(delta_info_broadcast, delta_info);
				}

				current_bag = updated_bag;
				delta_change = false;
			}
			__syncwarp();
			/////////////////////////
			////////////////////////
			unsigned total_avail_tb = 0;
			//find the available tbs
			for (unsigned i = get_lane_id(); i < num_tb_32; i += 32) {
				unsigned m_tb = (last_assigned_tb + i) % num_tb_32;
				unsigned m_status = cub::ThreadLoad<cub::LOAD_CG>(&(regular_status[m_tb]));
				unsigned regular_seq = rs_get_seq(m_status);
				bool avail_tb = (regular_seq != local_seq[m_tb]);
				unsigned avail_tb_mask = __ballot_sync(FULL_MASK, avail_tb);
				unsigned tb_count = count_bit(avail_tb_mask);
				unsigned m_idx = count_bit(set_bits(avail_tb_mask, 0, get_lane_id(), 32));
				//write to tb list
				if (avail_tb) {
					tb_list[total_avail_tb + m_idx] = m_tb;
					new_work[m_tb] = 0;
				}
				total_avail_tb += tb_count;
				tb_status[m_tb] = m_status;
			}

			if (probe_low_phase) {
				//clip to 1
				total_avail_tb = min(total_avail_tb, 1);
			}

			//find available bags
			unsigned bag_id_real = current_bag + get_lane_id();
			unsigned bag_id_map = bag_id_real % NUM_BAG;
			unsigned m_avail_work = 0;
			if (get_lane_id() < concur_bag) {
				unsigned m_assign_ptr = assign_ptr[bag_id_map];
				unsigned m_read_ptr = read_ptr[bag_id_map];
				m_avail_work = m_read_ptr - m_assign_ptr;
				if (m_assign_ptr > m_read_ptr) {
					//MAX unsigned case
					m_avail_work = (UINT32_MAX - m_assign_ptr) + m_read_ptr;
				}
			}

			//add them up
			unsigned total_avail_work = WarpReduce(temp_storage).Sum(m_avail_work);
			total_avail_work = __shfl_sync(FULL_MASK, total_avail_work, 0);

			if ((total_avail_tb > 0) && (total_avail_work > 0)) {
				//decide assignment granularity
				unsigned per_warp_work = total_avail_work / (total_avail_tb * num_warp);
				per_warp_work = max(per_warp_work, mini_grain);
				//round up to the next pow 2
				if (count_bit(per_warp_work) != 1) {
					unsigned next_bit = find_ms_bit(per_warp_work) + 1;
					per_warp_work = set_bits(0, FULL_MASK, next_bit, 1);
				}
				per_warp_work = min(per_warp_work, 32);
				unsigned bit_pos = find_ms_bit(per_warp_work);
				unsigned per_tb_work = per_warp_work * num_warp;

				//assign to tbs
				unsigned cur_assigned_tb = 0;
				while (cur_assigned_tb != total_avail_tb) {
					//check concurrent bags
					unsigned m_num_work = 0;
					unsigned m_translated_assign_ptr;
					unsigned m_assign_ptr = assign_ptr[bag_id_map];
					if (get_lane_id() < concur_bag) {
						unsigned m_read_ptr = read_ptr[bag_id_map];
						unsigned m_read_bk = m_read_ptr / BLOCK_SIZE;
						unsigned m_assign_bk = m_assign_ptr / BLOCK_SIZE;
						unsigned end_lower_ptr = BLOCK_SIZE;
						if (m_assign_bk == m_read_bk) {
							end_lower_ptr = m_read_ptr % BLOCK_SIZE;
						}
						unsigned m_assign_lower_ptr = m_assign_ptr % BLOCK_SIZE;
						unsigned cache_idx = m_assign_bk % MB_CACHE_SIZE;
						unsigned entry = manager_bk_cache[bag_id_map][cache_idx];
						unsigned translated_upper_ptr = bc_get_idx(entry) * BLOCK_SIZE;
						//will be shuffled
						m_num_work = end_lower_ptr - m_assign_lower_ptr;
						m_translated_assign_ptr = translated_upper_ptr + m_assign_lower_ptr;

					}
					if (probe_low_phase) {
						//cap to low
						m_num_work = min(m_num_work, probe_low_work);
					}
					//Available bags
					bool availbe_bag = (m_num_work > 0);
					unsigned availbe_bag_mask = __ballot_sync(FULL_MASK, availbe_bag);
					//try to exhaust either available TB or
					if (availbe_bag_mask != 0) {
						//assign the highest available bag
						unsigned bag_lane = find_nth_bit(availbe_bag_mask, 0, 1);
						unsigned num_work = __shfl_sync(FULL_MASK, m_num_work, bag_lane);
						unsigned m_idx = get_lane_id();
						unsigned m_offset = m_idx * per_tb_work;
						unsigned m_size = per_tb_work;
						if ((m_offset + m_size) > num_work) {
							//the last one
							m_size = num_work - m_offset;
						}
						unsigned real_assign_ptr = __shfl_sync(FULL_MASK, m_translated_assign_ptr, bag_lane);
						unsigned m_real_ptr = real_assign_ptr + m_offset;
						bool valid = ((cur_assigned_tb + m_idx) < total_avail_tb);
						valid = (m_offset < num_work) && valid;
						if (valid) {
							//set tb assignment buffers
							unsigned m_tb = tb_list[cur_assigned_tb + m_idx];
							unsigned long long m_assignment = agm_set_real_ptr(0, m_real_ptr);
							m_assignment = agm_set_size(m_assignment, m_size);
							m_assignment = agm_set_grain_bit(m_assignment, bit_pos);
							m_assignment = agm_set_bag_id(m_assignment, (current_bag + bag_lane) % NUM_BAG);
							unsigned seq = (local_seq[m_tb]) ^ 1;
							m_assignment = agm_set_seq(m_assignment, seq);
							local_seq[m_tb] = seq;
							//get tb from lst
							cub::ThreadStore<cub::STORE_CG>(&(agm_buf[m_tb]), m_assignment);
							new_work[m_tb] = m_size;

						}

						//update assign ptr, from the last valid assignment
						unsigned last_lane = find_ms_bit(__ballot_sync(FULL_MASK, valid));
						unsigned assigned_work = __shfl_sync(FULL_MASK, m_offset + m_size, last_lane);
						//the bag lane will update it
						if (get_lane_id() == bag_lane) {
							assign_ptr[bag_id_map] = m_assign_ptr + assigned_work;
						}

						//update available tb
						cur_assigned_tb += (last_lane + 1);
						//gather stats
						{
							prof_assigned_tb += (last_lane + 1);
							prof_assigned_work += assigned_work;
						}
					} else {
						break;		// no bag available
					}
				}
				//set last tb for rr,
				if (cur_assigned_tb == total_avail_tb) {
					//set to total - 1 tb's next tb
					unsigned tb = tb_list[total_avail_tb - 1];
					last_assigned_tb = (tb + 1) % num_tb_32;
				} else {
					//start from last assigned+1 tb
					unsigned tb = tb_list[cur_assigned_tb];
					last_assigned_tb = tb;
				}
			}

			__syncwarp();

		} else if ((updated_bag - current_bag) > (NUM_BAG * 2)) {
			//exit
			manager_exit = 1;
			for (int m_tb = get_lane_id(); m_tb < num_tb_real; m_tb += 32) {
				cub::ThreadStore<cub::STORE_CG>(&(agm_buf[m_tb]), EXIT);
			}
		}

	}

/////////////
/////////
}

/*
 * sender stuff
 */

///////////////////////////
__device__ __forceinline__ unsigned fc_get_cur_count(unsigned fc_entry) {
	return extract_bits(fc_entry, 0, 16);
}

__device__ __forceinline__ unsigned fc_get_size(unsigned fc_entry) {
	return extract_bits(fc_entry, 16, 16);
}

__device__ __forceinline__ unsigned fc_set_cur_count(unsigned fc_entry, unsigned cur_count) {
	return set_bits(fc_entry, cur_count, 0, 16);
}

__device__ __forceinline__ unsigned fc_set_size(unsigned fc_entry, unsigned size) {
	return set_bits(fc_entry, size, 16, 16);
}

__shared__ unsigned tb_lock;
__shared__ unsigned long long tb_agm_local;
__shared__ unsigned free_counter_idx_local;
__shared__ unsigned assign_rr;
__shared__ unsigned free_counter[8];
__shared__ unsigned long long warp_agm[32];
__shared__ unsigned delta_info_local;
__shared__ unsigned regular_local_seq;
__shared__ unsigned regular_assigned_work;
__shared__ unsigned regular_processing_work;
#define BK_CACHE_SIZE 4
__shared__ unsigned bK_cache_lock[NUM_BAG];
__shared__ unsigned bk_cache[NUM_BAG][BK_CACHE_SIZE];

#define TB_COOP_MUL 4
#define MIN_TB_COOP_PER_WARP_WORK 64
#define MIN_TB_COOP_WARP 2
__shared__ unsigned tb_coop_lock;
__shared__ unsigned tb_coop_vertex_id[32];
__shared__ unsigned tb_coop_edge[32];
__shared__ unsigned long long tb_coop_oringinal_agm[32];
__shared__ unsigned tb_coop_work_size[32];

__device__ void worklist::init_regular() {

	if (threadIdx.x < 32) {
		if (get_lane_id() == 0) {
			tb_lock = 0;
			tb_agm_local = 0;
			assign_rr = 0;
			regular_local_seq = 1;
			regular_assigned_work = 0;
			regular_processing_work = 0;
			tb_coop_lock = 0;
		}
		if (get_lane_id() < 8) {
			free_counter[get_lane_id()] = 0;
		}

		for (int bag_id = get_lane_id(); bag_id < NUM_BAG; bag_id += 32) {
			bK_cache_lock[bag_id] = 0;
			for (int i = 0; i < BK_CACHE_SIZE; i++) {
				bk_cache[bag_id][i] = NOT_FOUND;
			}
		}

		if (get_lane_id() < num_warp) {
			warp_agm[get_lane_id()] = 0;
			tb_coop_work_size[get_lane_id()] = 0;
		} else {
			warp_agm[get_lane_id()] = 1;
			tb_coop_work_size[get_lane_id()] = 1;
		}
	}

	__syncthreads();
}

__device__ void worklist::tb_coop_process(CSRGraph& graph, int warp_id) {
	unsigned work_size = tb_coop_work_size[warp_id];
	if (work_size > 0) {
		unsigned edge_start = tb_coop_edge[warp_id];
		unsigned vertex_id = tb_coop_vertex_id[warp_id];
		unsigned long long coop_assignment = tb_coop_oringinal_agm[warp_id];
		unsigned src_bag_id = agm_get_bag_id(coop_assignment);
		for (int offset = get_lane_id(); offset < work_size; offset += 32) {
			index_type edge = edge_start + offset;
			index_type dst = graph.edge_dst[edge];
			edge_data_type wt = graph.edge_data[edge];
			node_data_type new_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[vertex_id])) + wt;
			node_data_type dst_dist = cub::ThreadLoad<cub::LOAD_CG>(&(graph.node_data[dst]));
			if (dst_dist > new_dist) {
				atomicMin(&(graph.node_data[dst]), new_dist);
				unsigned dst_bag_id = dist_to_bag_id_int(src_bag_id, new_dist);
				push_work(dst_bag_id, dst);
			}
		}

		__syncwarp();
		epilog(coop_assignment, 1, true);

		//reset
		if (get_lane_id() == 0) {
			tb_coop_work_size[warp_id] = 0;
		}
	}
}

__device__ __forceinline__ void worklist::tb_coop_assign(int vertex_id, int first_edge, unsigned long long m_assignment, int& m_size, unsigned tb_coop_lane) {
	bool success = false;
	unsigned lock_val;
	if (get_lane_id() == 0) {
		lock_val = atomicCAS(&tb_coop_lock, 0, 1);
	}
	lock_val = __shfl_sync(FULL_MASK, lock_val, 0);
	if (lock_val == 0) {
		//grabbed lock, do the assignment
		//find free buffers
		bool free_buffer = (tb_coop_work_size[get_lane_id()] == 0);
		unsigned free_mask = __ballot_sync(FULL_MASK, free_buffer);
		unsigned free_count = count_bit(free_mask);
		if (free_count > MIN_TB_COOP_WARP) {
			success = true;
			unsigned m_offset = count_bit(set_bits(free_mask, 0, get_lane_id(), 32));
			unsigned leader_vertex_id = __shfl_sync(FULL_MASK, vertex_id, tb_coop_lane);
			unsigned leader_first_edge = __shfl_sync(FULL_MASK, first_edge, tb_coop_lane);
			unsigned leader_size = __shfl_sync(FULL_MASK, m_size, tb_coop_lane);

			unsigned per_warp_work = (leader_size / free_count) + 1;
			//round up to 32
			per_warp_work = roundup(per_warp_work, 32);
			per_warp_work = max(MIN_TB_COOP_PER_WARP_WORK, per_warp_work);

			//calc my assignment
			unsigned m_start = per_warp_work * m_offset;
			unsigned m_end = min(leader_size, m_start + per_warp_work);

			bool valid = false;
			if ((m_end > m_start) && free_buffer) {
				//in range work
				tb_coop_vertex_id[get_lane_id()] = leader_vertex_id;
				tb_coop_edge[get_lane_id()] = leader_first_edge + m_start;
				tb_coop_oringinal_agm[get_lane_id()] = m_assignment;
				__threadfence_block();
				//making sure mem consistency
				tb_coop_work_size[get_lane_id()] = m_end - m_start;
				valid = true;
			}
			unsigned valid_count = count_bit(__ballot_sync(FULL_MASK, valid));
			if (get_lane_id() == 0) {
				unsigned counter_idx = agm_get_grain_bit(m_assignment);
				//this will be decreased by coop workers also
				atomicAdd(&(free_counter[counter_idx]), valid_count);
			}
		}
		if (get_lane_id() == 0) {
			__threadfence_block();
			atomicExch(&tb_coop_lock, 0);
		}
	}

	//on success, set msize to 0, so it is not processed by other methods
	if (success && (tb_coop_lane == get_lane_id())) {
		m_size = 0;
	}
	//else do nothing process as normal
}

__device__ void worklist::get_global(unsigned warp_id) {
//poll global status
	unsigned lock_val;
	if (get_lane_id() == 0) {
		lock_val = atomicCAS(&tb_lock, 0, 1);
	}
	lock_val = __shfl_sync(FULL_MASK, lock_val, 0);
	if (lock_val == 0) {
		unsigned delta_info;
		if (get_lane_id() == 0) {
			delta_info = cub::ThreadLoad<cub::LOAD_CG>(delta_info_broadcast);
		}
		__syncwarp();
//we have the right to check global status
//find out how many warps are availble
//use rr for load balancing
		unsigned rr = assign_rr;
		unsigned m_warp = (get_lane_id() + rr) % 32;
		bool warp_avail = (warp_agm[m_warp] == 0);
		unsigned avail_mask = __ballot_sync(FULL_MASK, warp_avail);
		unsigned num_avail = count_bit(avail_mask);
		unsigned m_idx = count_bit(set_bits(avail_mask, 0, get_lane_id(), 32));
//prevent race condition
//it is possible to have non-availble warp (this warp can be assigned while here)
		if (num_avail > 0) {
			//check cached status first
			unsigned long long tb_assginment = tb_agm_local;
			unsigned free_counter_idx = free_counter_idx_local;
			if (tb_assginment == 0) {
				//check global
				if (get_lane_id() == 0) {
					tb_assginment = cub::ThreadLoad<cub::LOAD_CG>(&(agm_buf[block_id_x()]));
				}
				//find free counter
				bool has_free_counter = false;
				if (get_lane_id() < 8) {
					has_free_counter = (free_counter[get_lane_id()] == 0);
				}
				unsigned vote_result = __ballot_sync(FULL_MASK, has_free_counter);
				free_counter_idx = find_ms_bit(vote_result);
				tb_assginment = __shfl_sync(FULL_MASK, tb_assginment, 0);
				//see if seq matches with local seq
				unsigned global_seq = agm_get_seq(tb_assginment);
				unsigned local_seq = regular_local_seq;

				if (tb_assginment == EXIT) {
					//exit procedure
					warp_agm[get_lane_id()] = EXIT;
					//this ensures 0 work
					tb_assginment = 0;
				} else {
					if ((free_counter_idx != NOT_FOUND) && (global_seq == local_seq)) {
						//there is work if match
						//check exit procedure
						if (get_lane_id() == 0) {
							//can write new global, let the MTB know
							regular_local_seq = (local_seq ^ 1);						//xor 1
							unsigned size = agm_get_size(tb_assginment);
							atomicAdd(&regular_assigned_work, size);
							//set counter
							unsigned fc_entry = 0;
							fc_entry = fc_set_cur_count(fc_entry, size);
							fc_entry = fc_set_size(fc_entry, size);
							free_counter[free_counter_idx] = fc_entry;
						}
					} else {
						tb_assginment = 0;
					}
				}
			}
			unsigned tb_ptr = agm_get_real_ptr(tb_assginment);
			unsigned grain_bit = agm_get_grain_bit(tb_assginment);
			unsigned grain = set_bits(0, FULL_MASK, grain_bit, 1);
			unsigned total_size = agm_get_size(tb_assginment);
			unsigned m_offset = m_idx * grain;

			bool valid = (warp_avail && (m_offset < total_size));
			unsigned valid_mask = __ballot_sync(FULL_MASK, valid);
			if (valid) {
				unsigned m_size = grain;
				unsigned m_ptr = tb_ptr + m_offset;
				unsigned m_next_offset = m_offset + grain;
				if (get_lane_id() == find_ms_bit(valid_mask)) {
					//the last lane set the local buff
					if (m_next_offset >= total_size) {
						m_size = total_size - m_offset;
						//reset buf status
						tb_agm_local = 0;
					} else {
						tb_assginment = agm_set_real_ptr(tb_assginment, tb_ptr + m_next_offset);
						tb_assginment = agm_set_size(tb_assginment, total_size - m_next_offset);
						tb_agm_local = tb_assginment;
						free_counter_idx_local = free_counter_idx;
					}
					assign_rr = (rr + get_lane_id() + 1) % 32;
				}

				//assign to warp
				unsigned long long m_assignment = agm_set_real_ptr(tb_assginment, m_ptr);
				m_assignment = agm_set_size(m_assignment, m_size);
				//use grain bit as counter idx
				m_assignment = agm_set_grain_bit(m_assignment, free_counter_idx);
				warp_agm[m_warp] = m_assignment;
			}
		}

		if (get_lane_id() == 0) {
			delta_info_local = delta_info;
			//see if there is need to change status
			unsigned m_status = rs_set_seq(0, regular_local_seq);
			m_status = rs_set_processing(m_status, regular_processing_work);
			m_status = rs_set_assigned(m_status, regular_assigned_work);
			//change global status
			cub::ThreadStore<cub::STORE_CG>(&(regular_status[block_id_x()]), m_status);

			//unlock
			__threadfence_block();
			atomicExch(&tb_lock, 0);
		}
		__syncwarp();
	}
}

__device__ __forceinline__ unsigned worklist::get_assignment(CSRGraph& graph, unsigned long long &m_assignment, unsigned warp_id, unsigned* work_count, unsigned total_work) {
	while (1) {

		//do tb coop
		tb_coop_process(graph, warp_id);

//check my assignment
		if (get_lane_id() == 0) {
			m_assignment = warp_agm[warp_id];
		}
		m_assignment = __shfl_sync(FULL_MASK, m_assignment, 0);
		if (m_assignment == 0) {
			//poll global status
			get_global(warp_id);
		} else if (m_assignment == EXIT) {
			//exit procedure
			int tid = thread_id_x() + block_dim_x() * block_id_x();
			work_count[tid] = total_work;
			asm volatile("exit;");
		} else {
			//there is actual work assigned to me
			//reset
			unsigned num = agm_get_size(m_assignment);
			if (get_lane_id() == 0) {
				warp_agm[warp_id] = 0;
				atomicSub(&regular_assigned_work, num);
				atomicAdd(&regular_processing_work, num);
			}
			__syncwarp();
			return num;
		}
	}
}
__device__ __forceinline__ void worklist::update_bk_cache(unsigned bag_id, unsigned m_tag, unsigned cache_idx) {
//try update
	if (atomicCAS(&(bK_cache_lock[bag_id]), 0, 1) == 0) {
		unsigned bk_array_idx = m_tag & (num_block_linear - 1);	//Virtual bk idx
		unsigned real_ptr = cub::ThreadLoad<cub::LOAD_CG>(&(bk_ptr_array[bag_id][bk_array_idx]));
		if (real_ptr != NOT_FOUND) {
			//now we can update the cache
			unsigned real_idx = real_ptr / BLOCK_SIZE;
			unsigned entry = 0;
			entry = bc_set_idx(entry, real_idx);
			entry = bc_set_tag(entry, m_tag);
			//write to cache
			bk_cache[bag_id][cache_idx] = entry;
		}
		__threadfence_block();
//unlock
		atomicExch(&bK_cache_lock[bag_id], 0);
	}
}

__device__ __forceinline__ unsigned worklist::translate_write_ptr(unsigned bag_id, unsigned m_ptr) {
	unsigned m_tag = m_ptr / BLOCK_SIZE;
	unsigned cache_idx = m_tag % BK_CACHE_SIZE;
	do {
		unsigned entry = bk_cache[bag_id][cache_idx];
		unsigned cur_tag = bc_get_tag(entry);
		if (cur_tag != m_tag) {
			//try update
			update_bk_cache(bag_id, m_tag, cache_idx);
		} else {
			unsigned real_bk_idx = bc_get_idx(entry);
			unsigned write_ptr = real_bk_idx * BLOCK_SIZE + (m_ptr % BLOCK_SIZE);
			return write_ptr;
		}
	} while (1);
}

__device__ __forceinline__ void worklist::prefetch_bk_cache(unsigned bag_id, unsigned m_ptr) {
	unsigned prefetch_tag = (m_ptr + BLOCK_SIZE) / BLOCK_SIZE;
	unsigned cache_idx = prefetch_tag % BK_CACHE_SIZE;
	unsigned entry = bk_cache[bag_id][cache_idx];
	unsigned cur_tag = bc_get_tag(entry);
	if (cur_tag != prefetch_tag) {
//try update
		update_bk_cache(bag_id, prefetch_tag, cache_idx);
	}
}

__device__ __forceinline__ void worklist::push_work(unsigned bag_id, unsigned node) {
	unsigned active_mask = __activemask();
	unsigned match_mask = __match_any_sync(active_mask, bag_id);
	unsigned num = count_bit(match_mask);
	unsigned m_idx = count_bit(set_bits(match_mask, 0, get_lane_id(), 32));
	unsigned leader = find_ms_bit(match_mask);
	unsigned r_ptr;
	if (get_lane_id() == leader) {
		r_ptr = atomicAdd(&(resv_ptr[bag_id]), num);
	}
	r_ptr = __shfl_sync(active_mask, r_ptr, leader);
	unsigned m_ptr = r_ptr + m_idx;
//translate to actual address
	unsigned write_ptr = translate_write_ptr(bag_id, m_ptr);
	unsigned m_seg = (write_ptr / SEG_SIZE);
//set the done counter
	unsigned seg_mask = __match_any_sync(active_mask, m_seg);
	leader = find_ms_bit(seg_mask);
	cub::ThreadStore<cub::STORE_CG>(&(wl_data[write_ptr]), node);
	__threadfence();
	if (get_lane_id() == leader) {
		unsigned count = count_bit(seg_mask);
		atomicAdd(&(write_done[m_seg]), count);
//prefetch
		prefetch_bk_cache(bag_id, m_ptr);
	}
}

__device__ __forceinline__ void worklist::epilog(unsigned long long m_assignment, unsigned work_size, bool coop) {
	unsigned bag_id = agm_get_bag_id(m_assignment);
//try free local
	unsigned real_bk_idx = agm_get_real_bk_idx(m_assignment);
//grain bit is used as counter idx
	unsigned counter_idx = agm_get_grain_bit(m_assignment);
	if (get_lane_id() == 0) {
		unsigned fc_entry = atomicSub(&(free_counter[counter_idx]), work_size) - work_size;
		if (!coop) {
			atomicSub(&regular_processing_work, work_size);
		}
		unsigned cur_count = fc_get_cur_count(fc_entry);
		if (cur_count == 0) {
			//get assignment size
			unsigned size = fc_get_size(fc_entry);
			atomicAdd(&(done_work_count[bag_id]), size);
			__threadfence_block();
			free_counter[counter_idx] = 0;
			atomicAdd(&(read_done[real_bk_idx]), size);
		}
	}

	__syncwarp();
}
__device__ __forceinline__ unsigned worklist::pop_work(unsigned idx) {
	return cub::ThreadLoad<cub::LOAD_CG>(&(wl_data[idx]));
}

__device__ __forceinline__ unsigned worklist::dist_to_bag_id_int(unsigned bag_id, node_data_type dst_dist) {
	unsigned delta_info = delta_info_local;
	int cur_shifted = bfg_get_dist(delta_info);
	unsigned delta_shift = bfg_get_delta(delta_info);
	unsigned cur_bag = bfg_get_bag(delta_info);
	int dst_shifted = dst_dist >> delta_shift;
	int delta_offset = max(0, dst_shifted - cur_shifted);	//clip to 0
	delta_offset = min(NUM_BAG - 1, delta_offset);	//clip to NUM_BAG-1
	if ((delta_offset == 0) && (cur_bag != bag_id)) {
//return (cur_bag + 1) % NUM_BAG;
		return (cur_bag + 1) % NUM_BAG;
	} else {
		return (cur_bag + delta_offset) % NUM_BAG;
	}
}

#endif /* WL_H_ */
