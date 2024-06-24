The idea is to use loss functions to try to make filters as similar as possible between layers.

This can in turn let us more efficient network representations - we could index existing weights and thus save memory in network binary.

The first step encapsulates doing this for squeezenet in order to verify feasibility, if it works, we can construct architectures that allow this in a better way.

![Filter sharing idea image](Filter_sharing.png)

After 60 epochs, we can see that the filters start getting fairly similar:

(Pdb) reshaped_filters[0]
tensor([[ 0.1094, -0.0571, -0.0893],
        [ 0.0303, -0.0831,  0.1064],
        [-0.0806,  0.0383, -0.0532]], device='cuda:0',
       grad_fn=<SelectBackward0>)
(Pdb) selected_kernels[0]
tensor([[ 0.0850, -0.0314, -0.0943],
        [-0.0011, -0.0600,  0.0897],
        [-0.0853,  0.0577, -0.0068]], device='cuda:0',
       grad_fn=<SelectBackward0>)

At this point we have the following test accuracy:

Test Loss: 1.5494785368442536, Test Accuracy: 72.76%

#TODO

* Add specific optimizer for the specific filter parameters