
The nbody-gpu takes 6 args

an integer N to specify N random particles or a tsv file to load a state or the word planet to get the solar system.

dt being an amount of time change between each state (int)

Steps which is how many times you simulate (int)

dump_every Which allows us to specify when to dump the state into an output (int)

CudaBlockSize Which is the how large a block will be when put on our device.

and output which tells the function where to dump the output. (string)

nbody <N|input.tsv> <dt> <steps> <dump_every> <CudaBlockSize>[output.tsv]

to compile run make

to clean run make clean

the same goes for sequential execept we dont use the nbthreads arg

also to make the file use make nbody

When benched we get this time:

Time: 131.087 [100,000] parts
Block size: 128

Time: 2.433 [10,000] parts
Block size: 128

Time: 0.133687 [1,000] parts
Block size: 128

Time: 144.311 [100,000] parts
Block size: 256

Time: 2.44059 [10,000] parts
Block size: 256

Time: 0.252167 [1,000] parts
Block size: 256

Time: 132.447 [100,000] parts
Block size: 64

Time: 1.84222 [10,000] parts
Block size: 64

Time: 0.101751 [1,000] parts
Block size: 64

Time: 3347.83 Number of threads: 1 (seq)s [100,000] parts
Time: 32.3903 Number of threads: 1 (seq)s [10,000] parts
Time: 0.326239 Number of threads: 1 (seq)s [1,000] parts

We can see that with the GPU and even with varying block size we get extream speedup.
~30x compared to sequential. You can see when we make the block size larger it takes
a bit more time to create those blocks and we see on 1000 particles the overhead we
pay for that. something that suprised me was that we still were seeing speedup with
1000 particles I thought by then it would have more overhead than sequential and would have taken a dip.
