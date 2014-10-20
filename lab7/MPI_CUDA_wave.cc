/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 
 * 
 * To be completed by students as Assignment 7 of CS 179
 */


#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>

//for MPI
#include <mpi.h>


#include <cuda_runtime.h>
#include <algorithm>

#include "Cuda1DFDWave_cuda.cuh"



int main(int argc, char **argv){
    
    //making sure output directory exists
    std::ifstream test("output");
    if (!(bool)test){
        printf("Need output directory. Aborting.");
        exit(10);
    }
    
    
    /* Parameters for the assignment and simulation (similar to Assignment 6) */

    bool writeFiles = true;
    
    const size_t numberOfIntervals = 1e5;
    const size_t numberOfTimesteps = 3e5;
    const size_t numberOfOutputFiles = 30;
    size_t TIMESTEP_COMMUNICATION_INTERVAL = 10;
    
    const float courant = 1.0;
    const float omega0 = 10;
    const float omega1 = 100;
    
    
    const size_t numberOfNodes = numberOfIntervals + 1;
    const float courantSquared = courant * courant;
    const float c = 1;
    const float dx = 1./numberOfIntervals;
    const float dt = courant * dx / c;
    
    

    /* Variables to store rank and the number of processes.
    They're set to default values, in case MPI isn't turned on */

    int rank = 0; 
    int numberOfProcesses = 1;


    /* TODO: Initialize MPI, and get this process's rank, as well
    as the total number of processes */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numberOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);


    

    if (numberOfProcesses == 1) {
        printf("Remember, to run this with mpi, you need to do "
           "\"mpirun -np X (your program and its arguments)\"\n");
    }

    //Check arguments
    //Note that MPI_Init filters out MPI-specific arguments
    if (argc < 3){
        //If message is bad, get the 0th process to tell us
        if (rank == 0){
            printf("Usage: (threads per block) (max number of blocks)\n");
        }

        //And exit
        MPI_Finalize();
        exit(-1);
    }


    /* Calculating blocks and threads per block */

    const unsigned int threadsPerBlock = atoi(argv[1]);
    const unsigned int maxBlocks = atoi(argv[2]);

    const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
                numberOfNodes/float(threadsPerBlock)));

    printf("threads blocks: %d %d\n", threadsPerBlock, blocks);
    
    //Preparing file index for output
    unsigned int fileIndex = 0;


/***********************************************************************/

/* TODO: Complete the implementation of the assignment. Your goal is
to build a program that divides the positions into intervals per
process, and uses CUDA to solve the wave equation on each interval,
and MPI to exchange necessary data between processes as the need arises.

Note that the rate of data exchange should be determined by
m = TIMESTEP_COMMUNICATION_INTERVAL steps above (i.e. you should exchange
data between processes every "m" timesteps.

In addition, set boundary conditions at every timestep, as we did
in Assignment 6.

There are a lot of ways to implement this assignment. In the comments
labeled "Suggested TODO", I outline one possible method - these
suggestions are meant to be helpful, but if you have a better way,
feel free to implement your own method. 

Note that, even if you choose not to follow the suggested TODOs, 
you still need to complete the assignment's specifications!
(e.g. setting boundary conditions is a "Suggested TODO" because
you may set them at a different place in your program, not because
setting boundary conditions is optional!)

*/



    /* Suggested TODO: Set up some important size parameters, such
    as the size of the array per process and the amount of
    memory you'll need for both your own data and the "redundancy
    intervals" of data passed to you by your adjacent processes. 

    Keep in mind: In case the number of processes doesn't divide
    the number of nodes equally, you'll need to have one of your
    processes absorb the "remaining" values (most likely the
    last process in your group, where last is the process with ID
    "numberOfProcesses - 1". */

    size_t m = TIMESTEP_COMMUNICATION_INTERVAL;

    int nodesPerProcess = numberOfNodes / numberOfProcesses;

    int totalSize;
    int sizeArray;
    printf("Total Number of Nodes: %d \n", numberOfNodes);

    int startNodeIndex;
    int endNodeIndex;

    if (rank < numberOfProcesses - 1) {
        sizeArray = nodesPerProcess;
        printf("Rank: %d  Size: %d \n", rank, sizeArray);
        printf("Rank: %d   from %d  to %d \n", rank, rank*sizeArray, (rank+1)*sizeArray-1);
        totalSize = sizeArray + 2 * m;

        startNodeIndex = rank*sizeArray;
        endNodeIndex = (rank+1)*sizeArray-1;

    } else if (rank == numberOfProcesses - 1) {
        // absorb remaining values
        sizeArray = numberOfNodes - rank*nodesPerProcess;
        printf("Rank: %d  Size: %d \n", rank, sizeArray);
        printf("Rank: %d   from %d  to %d \n", rank, rank*nodesPerProcess, rank*nodesPerProcess + sizeArray-1);
        totalSize = sizeArray + 2 * m;

        startNodeIndex = rank*nodesPerProcess;
        endNodeIndex = rank*nodesPerProcess + sizeArray-1;
    }

    

    /* Suggested TODO: Set up buffers for your communication with
    other processes. You'll probably want 4:
        - Buffer to send data to the right
        - Buffer to send data to the left
        - Buffer to receive data from the right
        - Buffer to receive data from the left
    */

    float send_buffer_left[m];
    float recv_buffer_left[m];
    float send_buffer_right[m];
    float recv_buffer_right[m];
    
    
    /* Suggested TODO: At this time, allocate memory on the GPU.
    You may want to allocate extra space in your GPU array(s) to
    account for the "redundancy intervals" that we'll be receiving.

    Since our initial conditions for this wave at time 0 will be
    0 everywhere, you can simply cudaMemset within your own process
    - no communication is required in this case, i.e. we don't have to
    receive an array of initial conditions from another source. */

    // allocate a single array of size 3 * totalSize, and zero out memory
    float *dev_Data;
    cudaMalloc((void**) &dev_Data, 3*totalSize * sizeof(float));
    cudaMemset(dev_Data, 0, 3*totalSize * sizeof(float));


    // need enough memory for each element, but not redundant data
    float *file_output_buffer = new float[sizeArray];


    /* Suggested TODO: Allocate memory for the buffer that you'll
    read into (from the GPU to the CPU) whenever you need to write
    a file. 

    (If you choose to do something different, you might have to change
    the file writing code that we give you.) */
    
    
    for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
            ++timestepIndex){
        
        if (timestepIndex % (numberOfTimesteps / 10) == 0) {
                printf("Process %d - Processing timestep %8zu (%5.1f%%)\n",
             rank, timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
        }


        /* Suggested TODO: If you take the approach of declaring one
        array on the GPU and treating it as three arrays,
        determine your old, current, and new addresses here.

        Note that these may be different from Assignment 6 if you decide
        to store your redundancy data on the GPU. */

        // adjust addresses for old, current, start
        int oldStart = totalSize * ((timestepIndex - 1) % 3);
        int currentStart = totalSize * ((timestepIndex) % 3);
        int newStart = totalSize * ((timestepIndex + 1) % 3);
 
        if (timestepIndex < 10) {
            printf("Process: %d  at time %d: oldStart: %d  currentStart: %d  newStart: %d \n", rank, timestepIndex, oldStart, currentStart, newStart);
        }

        
        /* Suggested TODO: At this time, complete the MPI implementation
        of data exchange. We need to send relevant current *and* old 
        data between the processes (this is different from assignment 6 
        because we're sending more than one value at a time, which
        necessitates that we send data for both arrays in order
        for our equation to be solvable. */


        if (timestepIndex % TIMESTEP_COMMUNICATION_INTERVAL == 0){
            
            /* Suggested TODO: We'll prepare to send and receive redundant
            intervals of *current* data. Declare all MPI_Request and MPI_Status
            objects for the four send/receive operations indicated
            previously. */
        
            MPI_Status status1;
            MPI_Request request1;
            MPI_Status status2;
            MPI_Request request2;
            MPI_Status status3;
            MPI_Request request3;
            MPI_Status status4;
            MPI_Request request4;

            /* Suggested TODO: If we're not the leftmost process:

                1. Copy the data we wish to send back from the GPU
                    (this will be our leftmost data that isn't in the
                    space where receive redundant data from the
                    process to our left).

                2. Request to send our data to the process to our left
                    (MPI_Isend). Remember to have tags!

                3. Request to receive the data that the process to our left 
                sends us (MPI_Irecv). Remember to have tags!

            */

            if (rank != 0) {

                // want to send data to the left!
                cudaMemcpy(send_buffer_left, &dev_Data[currentStart + m], m * sizeof(float), cudaMemcpyDeviceToHost);

                int tagSendLeft = 123 + rank;
                int tagReceiveLeft = 789 + rank;

                MPI_Isend(&send_buffer_left, m, MPI_FLOAT, rank-1, tagSendLeft, MPI_COMM_WORLD, &request1);

                MPI_Irecv(&recv_buffer_left, m, MPI_FLOAT, rank-1, tagReceiveLeft, MPI_COMM_WORLD, &request2);
            }

            
            /* Suggested TODO: Same idea as above, but this time,
            we want to copy our rightmost data, and send and receive
            to/from the process to our right (rank + 1).

            */

            if (rank != numberOfProcesses - 1) {
                cudaMemcpy(send_buffer_right, &dev_Data[currentStart + totalSize - 2 * m], m * sizeof(float), cudaMemcpyDeviceToHost);

                int tagSendRight = 789 + rank + 1;
                int tagReceiveRight = 123 + rank + 1;

                MPI_Isend(&send_buffer_right, m, MPI_FLOAT, rank+1, tagSendRight, MPI_COMM_WORLD, &request3);

                MPI_Irecv(&recv_buffer_right, m, MPI_FLOAT, rank+1, tagReceiveRight, MPI_COMM_WORLD, &request4);
            }

            
            /* Suggested TODO: Wait for the four MPI operations above to
            finish */

            if (rank != 0) {
                MPI_Wait(&request1,&status1);
                MPI_Wait(&request2,&status2);
            }
            if (rank != numberOfProcesses - 1) {
                MPI_Wait(&request3,&status3);
                MPI_Wait(&request4,&status4);
            }

            /* Suggested TODO: If we're not the leftmost process,
            copy the data we just received from the process to the left,
            back to the GPU in the place where we store our left=end 
            redundant data. */
            if (rank != 0) {
                cudaMemcpy(&dev_Data[currentStart], &recv_buffer_left, m*sizeof(float),
                    cudaMemcpyHostToDevice);
            }

            /* Suggested TODO: If we're not the rightmost process,
            same as the step above but with the right instead of the left. */
            if (rank != numberOfProcesses - 1) {
            cudaMemcpy(&dev_Data[currentStart + totalSize - m], &recv_buffer_right, m*sizeof(float),
                cudaMemcpyHostToDevice);
            }




            /* Suggested TODO: Repeat the six steps above, but with
            the "old" data instead of the "current" data.



            A possible alternative: If you want to shove everything
            into the calls above, you could read both old and current
            data into your buffer, send and receive it, and take it apart
            for copying back to the GPU. */

            MPI_Status status5;
            MPI_Request request5;
            MPI_Status status6;
            MPI_Request request6;
            MPI_Status status7;
            MPI_Request request7;
            MPI_Status status8;
            MPI_Request request8;

            if (rank != 0) {
                // want to send olddata to the left!
                cudaMemcpy(send_buffer_left, &dev_Data[oldStart + m], m * sizeof(float), cudaMemcpyDeviceToHost);

                int tagSendLeft = 456 + rank;
                int tagReceiveLeft = 555 + rank;

                MPI_Isend(&send_buffer_left, m, MPI_FLOAT, rank-1, tagSendLeft, MPI_COMM_WORLD, &request5);

                MPI_Irecv(&recv_buffer_left, m, MPI_FLOAT, rank-1, tagReceiveLeft, MPI_COMM_WORLD, &request6);
            }


            if (rank != numberOfProcesses - 1) {
                cudaMemcpy(send_buffer_right, &dev_Data[oldStart + totalSize - 2 * m], m * sizeof(float), cudaMemcpyDeviceToHost);

                int tagSendRight = 555 + rank + 1;
                int tagReceiveRight = 456 + rank + 1;

                MPI_Isend(&send_buffer_right, m, MPI_FLOAT, rank+1, tagSendRight, MPI_COMM_WORLD, &request7);

                MPI_Irecv(&recv_buffer_right, m, MPI_FLOAT, rank+1, tagReceiveRight, MPI_COMM_WORLD, &request8);
            }

            if (rank != 0) {
                MPI_Wait(&request5,&status5);
                MPI_Wait(&request6,&status6);
            }
            if (rank != numberOfProcesses - 1) {
                MPI_Wait(&request7,&status7);
                MPI_Wait(&request8,&status8);
            }

            /* Suggested TODO: If we're not the leftmost process,
            copy the data we just received from the process to the left,
            back to the GPU in the place where we store our left=end 
            redundant data. */
            if (rank != 0) {
                cudaMemcpy(&dev_Data[oldStart], &recv_buffer_left, m*sizeof(float),
                    cudaMemcpyHostToDevice);
            }

            /* Suggested TODO: If we're not the rightmost process,
            same as the step above but with the right instead of the left. */
            if (rank != numberOfProcesses - 1) {
            cudaMemcpy(&dev_Data[oldStart + totalSize - m], &recv_buffer_right, m*sizeof(float),
                cudaMemcpyHostToDevice);
            }
        
        }

        /* Suggested TODO: Call the kernel (or really, call the function
        that calls the kernel.) */
        kernelCall(dev_Data, oldStart, currentStart, newStart, courantSquared,
                        totalSize, blocks, threadsPerBlock);


        /* Suggested TODO: Set the left and right boundary conditions.
        Keep in mind that only the leftmost process will set the left
        boundary condition, and only the rightmost process will set 
        the right boundary condition. 

        Use the same boundary conditions as in Assignment 6.

        Remember that, depending on how you store your "redundant data"
        on the GPU, location (0) or (numberOfNodes - 1) may be offset
        from "index 0" or "index (numberOfNodes - 1)" of our "array"! */

        // set left boundary condition (a sum of sine waves) on the leftmost process
        if (rank == 0) {
            const float t = timestepIndex * dt;
            float left_boundary_value;
            if (omega0 * t < 2 * M_PI) {
                left_boundary_value = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
            } else {
                left_boundary_value = 0;
            }
            // offset by m (skip over redundant values)
            cudaMemcpy(&dev_Data[newStart + m], &left_boundary_value, sizeof(float),
                cudaMemcpyHostToDevice);
        }

        // set right boundary condition = 0 on the rightmost process
        // offset by -m, since we skip over redundant values
        if (rank == numberOfProcesses - 1) {
            cudaMemcpy(&dev_Data[newStart + totalSize - 1 - m], 0, sizeof(float),
                cudaMemcpyHostToDevice);
        }
        
        
        //If we need to write a file, prepare corresponding data
        //and put it in the file writing queue (master file thread will
        //handle it
        if (writeFiles == true && numberOfOutputFiles > 0 &&
                (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles)
                == 0) {
            

            /* TODO: Copy back our data into the file_output_buffer.
            Keep in mind that when we write our data to file, we don't
            want any redundant data. However, if we are the last process
            that absorbs "elements", we'll need to copy back those. */

            // Copy new data back from GPU to file_output on the CPU
            // offset by m, since we don't want redundant values.  note that sizeArray doesn't include
            // redundant values
            cudaMemcpy(file_output_buffer, &dev_Data[newStart+m], sizeArray * sizeof(float), cudaMemcpyDeviceToHost);

            //Determine file name
            
            printf("Writing output file - process %d, timestep %d\n", rank, timestepIndex);
            char *filename = new char[500];
            sprintf(filename, "output/displacements_%08zu_%03u.dat", 
                        fileIndex, rank);
            
            

            /* TODO: When we write our file, we write a position value,
            followed by the value of our function - the loop below
            increments this amount by dx at every iteration. 

            This assignment has each process write out its own file data
            individually. The script "makePlots2.py" eventually combines
            them and produces a plot.

            In assignment 6, we were able to start this loop effectively
            at 0, and continue to the end of our positions. However, 
            now our process must know when the x-values should start
            (e.g. if we have 10 processes, and 100000 nodes, process 5 needs
            to count starting at 50000, and eventually go to 59999.)

            Hence, calculate the index offset, i.e. where our nodes start. 
            (It depends on a value that you probably calculated 
            very early on) */


            int index_offset = startNodeIndex;

            int THREAD_DISP_SIZE = sizeArray;
            
            

            FILE* file = fopen(filename, "w");
            for (size_t nodeIndex = 0; nodeIndex < THREAD_DISP_SIZE; ++nodeIndex) {
                fprintf(file, "%e,%e\n", 
                    (index_offset + nodeIndex) * dx,
                    file_output_buffer[nodeIndex]);
            }
            fclose(file);
            
            fileIndex++;
        }
        
    }
    
    /* TODO: Finalize MPI environment */
    MPI_Finalize();
    return 0;
    
}