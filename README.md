# image-processing-hw1
In this assignment, we implemented the GrabCut algorithm to perform foreground segmentation on an input image. 

We replaced the background of the segmented object using Poisson blending, which is a powerful image editing technique that seamlessly blends the foreground object with a new background while preserving the object’s texture and color.

the implementation is based on the following articles:
http://vision.stanford.edu/teaching/cs231b_spring1415/papers/SIGGRAPH04_RotherKolmogorovBlake.pdf
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9246cbf8d7a489b2c6299317416c7dfdf747f72b

results for GrabCut:
![image](https://github.com/gayatalyosef/image-processing-hw1/assets/92454507/02509ff8-4c8f-4eb8-9f0e-7e5819c13e74)
![image](https://github.com/gayatalyosef/image-processing-hw1/assets/92454507/38352728-4bcc-495a-8376-a7ed11670dab)
![image](https://github.com/gayatalyosef/image-processing-hw1/assets/92454507/02e820dd-855e-40cd-989d-c49288640bcf)
