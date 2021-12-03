export eval_root=`pwd`

rm -rf *_final_dist
rm *_result

cd ads_int
make clean
cd $eval_root

cd ads_float
make clean
cd $eval_root

rm -rf nf_int 
rm -rf nf_float 

cd nv_int
make clean
cd $eval_root

cd nv_float
make clean
cd $eval_root

rm -rf cpu_sssp_int 
rm -rf cpu_sssp_float 