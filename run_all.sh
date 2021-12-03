#!/bin/bash

#function for run sssp
run_sssp () { 
	impl_name=$1
	input_list=$2
	input_path=$3 
	bin_path=$4
	#remove previous files if any
	rm -rf ${impl_name}_final_dist
	mkdir ${impl_name}_final_dist
	#read in graph file list, and run list (one line per graph)
	while read -r line
	do
		tokens=( $line );
		$bin_path -s ${tokens[1]} -o ${impl_name}_final_dist/${tokens[0]} ${input_path}/${tokens[0]};
	done < "$input_list"
}

#run this in eval_root directory
#the sssp impls output timing and work count results to file descriptor 3
#3>${impl_name}_result records results

#run int graph
input_list=./graph_list_int
input_path=./inputs/graph-int

#our solution 
impl_name=ads_int
bin_path=ads_int/sssp
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result

#nf
impl_name=nf_int
bin_path=./nf_int/build/lonestar/analytics/gpu/sssp/sssp-gpu
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result

#nv
impl_name=nv_int
bin_path=./nv_int/sssp
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result


#run float graph
input_list=./graph_list_float
input_path=./inputs/graph-float


#our solution 
impl_name=ads_float
bin_path=ads_float/sssp
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result


#nf
impl_name=nf_float
bin_path=./nf_float/build/lonestar/analytics/gpu/sssp/sssp-gpu
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result

#nv
impl_name=nv_float
bin_path=./nv_float/sssp
rm ${impl_name}_result
run_sssp $impl_name $input_list $input_path $bin_path 3>${impl_name}_result


#run cpu impl

#change this to match the number of hardware threads on the CPU
NUM_THREADS=20

run_ds () { 
	impl_name=$1
	input_list=$2
	input_path=$3 
	bin_path=$4
	#remove previous files if any
	rm -rf ${impl_name}_final_dist
	mkdir ${impl_name}_final_dist
	#read in graph file list, and run list (one line per graph)
	while read -r line
	do
		tokens=( $line );
		$bin_path ${input_path}/${tokens[0]} -noverify -algo=deltaStep -startNode=${tokens[1]} -delta=${tokens[2]} -t $NUM_THREADS; 
	done < "$input_list"
}

run_dj () { 
	impl_name=$1
	input_list=$2
	input_path=$3 
	bin_path=$4
	#remove previous files if any
	rm -rf ${impl_name}_final_dist
	mkdir ${impl_name}_final_dist
	#read in graph file list, and run list (one line per graph)
	while read -r line
	do
		tokens=( $line );
		$bin_path ${input_path}/${tokens[0]} -noverify -algo=dijkstra -startNode=${tokens[1]} -delta=${tokens[2]} -t 1; 
	done < "$input_list"
}
 
#run int graph
input_list=./cpu_graph_list_int
input_path=./inputs/graph-int

#run dj
impl_name=dj_int
bin_path=cpu_sssp_int/build/lonestar/analytics/cpu/sssp/sssp-cpu
rm ${impl_name}_result
run_dj $impl_name $input_list $input_path $bin_path 3>${impl_name}_result


#run ds 
impl_name=ds_int
bin_path=cpu_sssp_int/build/lonestar/analytics/cpu/sssp/sssp-cpu
rm ${impl_name}_result
run_ds $impl_name $input_list $input_path $bin_path 3>${impl_name}_result



#run float graph
input_list=./cpu_graph_list_float
input_path=./inputs/graph-float

#run dj
impl_name=dj_float
bin_path=cpu_sssp_float/build/lonestar/analytics/cpu/sssp/sssp-cpu
rm ${impl_name}_result
run_dj $impl_name $input_list $input_path $bin_path 3>${impl_name}_result


#run ds 
impl_name=ds_float
bin_path=cpu_sssp_float/build/lonestar/analytics/cpu/sssp/sssp-cpu
rm ${impl_name}_result
run_ds $impl_name $input_list $input_path $bin_path 3>${impl_name}_result

