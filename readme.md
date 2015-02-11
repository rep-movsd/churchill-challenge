Problem
-------
Given ten million points with float x, y and an int rank, id, find the 20 highest ranked points within a specified rectangle. You can setup any data structure you need beforehand and queries alone will be timed.

My rudimentary approach is described below

Initial setup
-------------

First we sort the points by Y coordinate


We subdivide the points recursively into two horizontal slices, with equal number of points

We adjust the split so that no two points with the same Y coordinate are in different slices.

We split 5 times recursively, getting 32 slices


Next we repeat this process for each slice, sorting by X and splitting it along the Y axis.

We split 5 times recursively, getting 32*32 chunks in total


Each of these chunks is sorted by rank, so that the best ranked points are the earliest.

We also calculate the actual bounding rectangle of each chunk.

For sanity, we store all rectangles as half open ranges : (lx, hx]


Next we order these chunks based on the best ranked point in each chunk.


For each chunk we make a vector of co-ordinates, and store each XY of each chunks points in the corresponding co-ord vector

We do this since we only need to read the X and Y while processing, and we can use SIMD to load 128 bits, or 4 floats

which is 2 XY co-ordinates, if we keep them next to each other


A chunk is only a pair of pointers into the array of input points that we copy.

Chunks contain no points values. We only make a copy of the input array because it's const and we need to sort it in

every which way.


Searching
---------

Let N be the number of top ranked points requested...


For each input rectangle, we check it against each of our chunks bounding rectangles.

We select those chunks that are completely inside the target rect and those chunks that overlap the rect


We have a result accumulator class - its job is to maintain the top N points.

When a new point is added to it, it checks if that point is in the top 20

If so it adds it and chucks out the 21st point if any

If not, it returns false, which means that any further points in the current chunk cannot possibly be a part

of the final N, because all of the following points in the chunk have a worse rank.



1) The chunks that are completely inside, we can process very quickly :

For each chunk, we test its first point's rank against the worst rank of the result accumulator.

If it is greater, then this whole chunk has no points of interest, so we skip it and go to the next

If not, we iterate through the chunks points and add them to the accumulator. If the accumulator returns false, then the

rest of the chunks points are not useful, so we can skip them and go to the next chunk.



2) The chunks that overlap the target rect, we do almost the same, except that when adding points we need to check if

each of the chunks points is actually within the target rectangle.


We check that as follows:

Load two XY coordinates into an SIMD register


Each X and Y of the co-ords is duplicated into two separate 128 bit SIMD registers

    X1Y1X2Y2 -> X1Y1X1Y1
    X1Y1X2Y2 -> X2Y2X2Y2

We put the rect's co-ord's into a 128 bit SIMD register, lets call it LLTTRRBB

The target rect has been adjusted outwards using _nextafterf() so that it is an open ranged rect rather than a closed one

Now we do a SIMD compare for both XYXYs using _mm_cmplt_ps

       X1 Y1 X1 Y1
    <  LL TT RR BB
       ===========
       A1 B1 C1 D1

if A1 and B1 are false, X and Y are both NOT less than (meaning >= ) the top and left : X >= LX && Y >= LY

if C1 and D1 are true , X and Y are both less than the rect bottom right : X < HX && Y < HY

We could not have done this comparison in 1 instruction if we did not convert the rect to an open ranged one


We check if A and B are false and C and D are true - if that holds, then the point is in the target rect and we can

feed it to the accumulator. We repeat the process for the second pair of coordinates.

The inner loop of this function is unrolled 3 times, so we process 6 points per iteration


The rest of the logic is the same as described above, we skip all the chunks remaining points if the accumulator

returned false anytime.



Rect testing

------------

We can do a similar SSE trick to test if one rect is completely inside another or if it overlaps as follows



Let us load one rectangles coordinates as a 128 bit value HX HY LX LY and the others as LX LY HX HY (which is the default layout for Rect) 

Both rectangles are half open range



Now if we do 

        HX1 HY1 LX1 LY1
    <=  LX2 LY2 HX2 HY2
        ---------------
        A   B   C   D


If  A and B are false but C and D are true, then the rectangles overlap

This is equivalent to 

    !(rcOther.hx <= rc.lx) &&
    !(rcOther.hy <= rc.ly) &&
    (rcOther.lx <= rc.hx) &&
    (rcOther.ly <= rc.hy);



Typically we could keep the HHLL of the target search rectangle loaded once for the entire search loop and inline everything



Then again, to test if one rectangle is completely in another:

Load an inclusive range rectangle RC1 in the 128 bit values HX1 HY1 LX1 LY1

Load an half open range rectangle RC2 in the 128 bit values HX2 HY2 LX2 LY2



        HX1 HY1 LX1 LY1
    >=  HX2 HY2 LX2 LY2
        ---------------
        A   B   C   D



If RC1 is completely within RC2, then A and B are false and C and D are true

This is equivalent to 

    !(rc1.hx >= rc2.hx) &&
    
    !(rc1.hy >= rc2.hy) &&
    
    (rc1.lx >= rc2.lx) &&
    
    (rc1.ly >= rc2.ly);



These techniques could easily be used with 16 bit integer co-ordinates too, to be able to test 8 pairs of rectangles for overlap in one shot!

Since we use the SSE unit for this, the processor hopefully continues to do other x86 stuff in parallel



Minor details
-------------

Keeping an 16 byte aligned point data structure seemed much faster at some point, so I stuck with it. I am not sure if it makes a big difference, but it seems like when the point is copied the compiler uses SSE to do it



The result accumulator uses an array of points, and uses binary search to find the place to insert the new candidate point.





Things that did not work:

-------------------------



For the result accumulator, I tried using a std::list, std::set and a priority heap using make_heap too.

I also tried storing pointers to points instead of actual points and I also tried storking ranks as int32_t alone. 

None worked as good as a dumb array.



I even tried an algorithm where I scan for second biggest element, put it at the end, and put the new point where the second biggest element was. This has only 3 assignments, but 20 iterations to find the position, per candidate point. This is still exactly the same performace as using lower_bound and an insertion as far as I could tell.



I tried splitting points on X and Y medians rather than at equal number of points

I tried splitting at the point where maximum gap occurred between points in the X and Y ranges

I tried a quad tree based approach and also a binary tree based approach, but didn't get better than this

I tried multithreading, with OpenMP and raw Win32 threads, but it seemed like waiting for the threads to finish took too long. 

I was testing on a VM and search was 5X slower even though I have an 8 core machine (with 6 cores given to the VM) and the setup phase was several times faster. It may have worked on a native machine.



I tried Boost R-trees, but it expects you to give it rectangles. You can give points and it will do its own quadratic split, but you have no access to the internal rectangles, so you cant sort them by rank as needed. The querying for overlapped or contained rects was extremely fast.



Before I even got the the level of reference.dll, I tried using sort by X and Y and binary search to find the rectangle spans. But this was terribly slow because it was quite possible the rectangle had millions of points.



I considered using stuff newer than SSE, but my system does not support those instructions.

In hindsight I ought to have developed on my new Haswell laptop with Windoes rather than on my old one with Linux and a VM

