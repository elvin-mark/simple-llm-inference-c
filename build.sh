cd build && \
    gcc -I../include -c ../src/**/*.c && \
    gcc -I../include -c ../main.c && \
    gcc *.o -lm -o main

cd ..
