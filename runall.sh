make clean
make
./bin/place -aux ./benchmark/ibm01/ibm01-cu85.aux | tee ibm01.log
./bin/place -aux ./benchmark/ibm02/ibm02-cu90.aux | tee ibm02.log
./bin/place -aux ./benchmark/ibm05/ibm05.aux | tee ibm05.log
./bin/place -aux ./benchmark/ibm07/ibm07-cu90.aux | tee ibm07.log
./bin/place -aux ./benchmark/ibm08/ibm08-cu90.aux | tee ibm08.log
./bin/place -aux ./benchmark/ibm09/ibm09-cu90.aux | tee ibm09.log
