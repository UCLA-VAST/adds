import os, sys, string
 
path1 = sys.argv[1]
path2 = sys.argv[2]

all_match = True

with open(sys.argv[3]) as graph_list_file:
	lines = graph_list_file.read().splitlines()
	for line in lines:
		graph = line.split('\t', 1 )[0]
		result_file_name1 = path1 + graph 
		result_file_name2 = path2 + graph 
		print(result_file_name1)
		print(result_file_name2)
		try: 
		    result_file1 = open(result_file_name1,"r")
		except:
		    print("File1 not found")
		    all_match = False
		try: 
		    result_file2 = open(result_file_name2,"r")
		except:
		    print("File2 not found")
		    all_match = False
		     
		
		while 1:
		    line1 = result_file1.readline()
		    line2 = result_file2.readline()
		    if (not line1) or (not line2):
		        if (line1):
		            print("path1 length not match")
		            print(line1)
		            all_match = False
		            break
		        elif (line2):
		            print("file2 length not match")
		            print(line2)
		            all_match = False
		            break
		        else:
		            break
		    else:
		        if line1 != line2:
		            print("not match")
		            print(line1, line2 , sep="", end="")
		            all_match = False
		            break
		print("match")

#end
if all_match:
	print("all results match!!!")
else:
	print("result mismatch!!!")

