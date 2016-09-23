import numpy as np
import os


def writeInDotFiles(in_list, debyeLen, coul_cut, seed_num, kappa):
	line_list = list(in_list)
	for i in range(len(line_list)):
		if ("pair_style	lj/cut/coul/debye" in line_list[i]):
			line_list[i] = 'pair_style	lj/cut/coul/debye {0} 2.5 {1}\n'.format(kappa, coul_cut)
		if ("velocity" in line_list[i]):
			line_list[i] = 'velocity all create 1.0 {0} dist gaussian\n'.format(seed_num*2)
 		if ("langevin" in line_list[i]):
 			line_list[i] = 'fix	2 all langevin 1.0 1.0 1.0 {0}\n'.format(seed_num)			
	try:
		os.remove(('in.polymer_new_debLen_'+debyeLen))
	except OSError:
		pass 			
	write_file = open(('in.polymer_new_debLen_'+debyeLen),'w')
	for i in range(len(line_list)):
	     write_file.write(line_list[i])
	write_file.close()
	name = 'in.polymer_new_debLen_'+debyeLen
	return name


def writeInDotFiles2(in_list, debyeLen, coul_cut, seed_num, kappa):
	line_list = list(in_list)
	for i in range(len(line_list)):
		if ("pair_style	lj/cut/coul/debye" in line_list[i]):
			line_list[i] = 'pair_style	lj/cut/coul/debye {0} 2.5 {1}\n'.format(kappa, coul_cut)
 		if ("langevin" in line_list[i]):
 			line_list[i] = 'fix	2 all langevin 1.0 1.0 1.0 {0}\n'.format(seed_num)
 		if ("read_data" in line_list[i]):
 			line_list[i] = 'read_data	data.polymer_debLen_'+debyeLen+'\n'		
	try:
		os.remove(('in.polymer_new2_debLen_'+debyeLen))
	except OSError:
		pass 			
	write_file = open(('in.polymer_new2_debLen_'+debyeLen),'w')
	for i in range(len(line_list)):
	     write_file.write(line_list[i])
	write_file.close()
	name = 'in.polymer_new2_debLen_'+debyeLen
	return name


def jobFileWriting(Lulz,argsListLeng):
	job_list = list(Lulz)
	for i in range(len(job_list)):
		if ("#$ -t 1-10" in job_list[i]):
			job_list[i] = "#$ -t 1-{0}\n".format(argsListLeng)
	try:
		os.remove("LAPHRON_ARGS_JOB.sh")
	except OSError:
		pass
	write_file3 = open("LAPHRON_ARGS_JOB.sh",'w')
	for i in range(len(job_list)):
	     write_file3.write(job_list[i])
	write_file3.close()


########################################################################################
num_points = int(raw_input("type the num of points make it 10 "))
saph_loop = raw_input("type the num of saphron loops, 10000 ")
mu = raw_input("type the value of mu, -4.0 ")

debMin = 0.1
debMax = 100

sample_file = 'in.polymer_new_template'
sample_in = open(sample_file,'r')
sample_lines = sample_in.readlines()
sample_in.close()

sample2_file = 'in.polymer_new2_template'
sample2_in = open(sample2_file,'r')
sample2_lines = sample2_in.readlines()
sample2_in.close()

sample_job_direc = 'LAPHRON_sh_template.sh'
read_in = open(sample_job_direc,'r')
job_lines = read_in.readlines()
read_in.close()

debyeLen_array = np.linspace(debMin,debMax,num_points)
print len(debyeLen_array)
seed_val= 946436

input_file_list = []
for x in debyeLen_array:
	debVal = "{0:.2f}".format(x)
	coulCut = "{0:.2f}".format(5*x)
	kappa = "{0:.2f}".format(float(1)/x)
	print debVal
	input_file_list.append(writeInDotFiles(sample_lines, debVal, coulCut, seed_val, kappa)+ " "+ saph_loop+ " "+ kappa +" "+ coulCut+ " "+ mu+" "+ debVal)
	writeInDotFiles2(sample2_lines, debVal, coulCut, seed_val, kappa)
	seed_val+=1


try:
	os.remove("args.list")
except OSError:
	pass
write_file5 = open("args.list",'w')
for i in range(len(input_file_list)):
     write_file5.write(input_file_list[i])
     write_file5.write("\n")
write_file5.close()

jobFileWriting(job_lines, len(input_file_list))
