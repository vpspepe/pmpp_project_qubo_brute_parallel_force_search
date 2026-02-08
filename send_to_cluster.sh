tar -czvf project.tar.gz data data_dense data_sparse src run.sh 
scp project.tar.gz vp51huji@lcluster7.hrz.tu-darmstadt.de:~/project.tar.gz
ssh vp51huji@lcluster19.hrz.tu-darmstadt.de << 'EOF'
    rm -rf project_directory
    rm -rf ~/.cache/lmod
    mkdir -p project_directory/
    tar -xzvf project.tar.gz -C project_directory/
    cd project_directory
    rm -rf build
    mkdir build
    module load cuda/12.5 gcc/13.1.0
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release  ../src
    make -j8
    cd ..
    rm out.txt
    sbatch run.sh
    until [ -f out.txt ]; do
        sleep 3
    done
EOF

scp  vp51huji@lcluster7.hrz.tu-darmstadt.de:~/project_directory/out.txt ./out.txt

