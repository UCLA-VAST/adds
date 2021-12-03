
all: sssp  

sssp: kernel.cu support.cu csr_graph.cu
	nvcc $^ -O3 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75  -rdc=true -lcudadevrt -Xptxas -O3  -Xptxas -v -lcuda -lcudart -o $@

clean:
	rm -f sssp *.o


