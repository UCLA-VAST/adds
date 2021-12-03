export eval_root=`pwd`

cd ads_float
make -j8
cd $eval_root

cd ads_int
make -j8
cd $eval_root

cd nv_float
make -j8
cd $eval_root

cd nv_int
make -j8
cd $eval_root

git clone https://github.com/IntelligentSoftwareSystems/Galois.git nf_int
cd nf_int
git checkout 38cd91cfb59a30cf0b4f7cf7d19f29d4d7188548
git apply ../nf_int.patch
git submodule init
git submodule update
mkdir build
cd build/
cmake .. -DGALOIS_CUDA_CAPABILITY="7.5" 
cd lonestar/analytics/gpu/sssp/
make -j8
cd $eval_root

git clone https://github.com/IntelligentSoftwareSystems/Galois.git nf_float
cd nf_float
git checkout 38cd91cfb59a30cf0b4f7cf7d19f29d4d7188548
git apply ../nf_float.patch
git submodule init
git submodule update
mkdir build
cd build/
cmake .. -DGALOIS_CUDA_CAPABILITY="7.5" 
cd lonestar/analytics/gpu/sssp/
make -j8
cd $eval_root
 
git clone https://github.com/IntelligentSoftwareSystems/Galois.git cpu_sssp_int
cd cpu_sssp_int
git checkout 38cd91cfb59a30cf0b4f7cf7d19f29d4d7188548
git apply ../cpu_sssp_int.patch 
mkdir build
cd build/
cmake .. 
cd lonestar/analytics/cpu/sssp
make -j8
cd $eval_root

git clone https://github.com/IntelligentSoftwareSystems/Galois.git cpu_sssp_float
cd cpu_sssp_float
git checkout 38cd91cfb59a30cf0b4f7cf7d19f29d4d7188548
git apply ../cpu_sssp_float.patch 
mkdir build
cd build/
cmake .. 
cd lonestar/analytics/cpu/sssp
make -j8
cd $eval_root