# curpas-ml

## Record

Before moving onto MNIST, I worked on XOR
* Had some explosions that caused neurons to either die or fire all the time
* Eventually got it stable, but it less learned and more slowly drifted randomly
* Switched from first activation to most activations in at timeframe, much more reliable and easier to handle
* Gave up on stochastic tracing and inhibitories and just did goold ol' gradients, took a bit of reading but after a few more explosiosn that solved xor well

Now MNIST
* This naive+unoptimised+cpu+generally bad approach to an SNN I've taken is very very very slow, so instead of waiting for 16 hours for it to finish I'm going to spend like 4 trying roll it into a slightly more logical layer structure.
* Didn't quite take four hours, but it is now way way faster.
* Running it again after adding a quick save function and cosine lr scheduler, and some extra thinking time during eval, 50 epochs should be plenty, hopefully it won't actively decrease in quality during overfitting since the data is so large, just not improve.
* It ended up fitting quite well 25 was around where it stopped improving. with an accuracy of 97.0%