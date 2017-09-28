# Recurrent Neural Nets input a vector and there are hidden states that output the tanh of the computation
# Just like sigmoid is a differentiable smooth function from 0 to 1, tanh is a differentiable and smooth function from -1 to 1
# The difference is that the current output of the RNN is fed back to itself as its input in the next instant
# Thus, the way to visualize is this- There is a vector input of size n and hidden layer of size m, then the effective input is some concatenation of the n and m sized arrays
# The net output is the softmax over the hidden state. Refer to the slides and presentation for extra clarity
# The adjustible parameter is the number of hidden state neurons, denoted by N. The input size and the output size are fixed
# Mathematically, X_eff[t] = X[t] | H[t - 1], where ' | ' represents concatenation pipe for generating the effective input
# Then, H[t] = tanh(X_eff[t].W_hidden[t] + B_hidden[t]), which is the computation for the next hidden state
# Then, the output is generated as-- Y[t]= softmax(H[t].W + B), which generates the output
# What is the input vector X? RNNs are usually used for time sequence modeling and hence, X is a feature representation of the input feature.
# As an instance, X can be one-hot encoding of words of a language or letters of an alphabet if we are dealing with NLP tasks using RNN
# What is the output vector Y? It is usually the feature representation of the desired output. 
# For instance, it can be the character that we wish to obtain from the previous sequence of character inputs.
# Note that the output, while training, can go wrong because of the wrongness of the weights as well as the wrongness of the state.
# Thus, WEIGHTS AND BIASES ARE SHARED ACROSS ALL THE UNFOLDED INSTANCES OF THE RNN
# We can go DEEP with RNNs as well!! We need to stack multiple hidden states or layers on top of each other, inside each instance, before the output is computed.
# There is a problem-- Empirically, it is observed that one-hot encoding is problematic (without embedding)
# Another problem-- RNNs can not model Long-Term Independencies over long unfolding lengths as the context from previous state is only a fixed length before.
# Thus, any task of reasonable practical utility demands the neural net to be deep in the sense of number of unfoldings it does. But deep nature has inherent problem of itself
# Problem with deep nature of RNN (or, in general, any NN)-- Vanishing/Exploding gradients cause divergence in nets (or, no convergence).
# The solution was invented to be LSTMs. These have gates and the gates can be used to remember or forget the context subject to learnt weights. This helps in modelling long-term dependencies.
# The gated nature can be shown to explain why gradients will not explode in the case of an LSTM.
# Mathematically (OH NO!!), we first concatenate the input with the context: X_eff[t] = X_t | H[t - 1]
# Then, there are three neural-network-like computations at three different gates: (with assumption that non-linear activation is sigmoid)
# Forget Gate: 
# f[t] = sigma(X_eff[t].W_f[t] + b_f[t]), where f is the forget gate output, W_f is the weight of forget gate, b_f is the bias of forget gate and sigma is non-linear activation
# Update Gate:
# u[t] = sigma(X_eff[t].W_u[t] + b_u[t]), where u is the update gate output, W_u is the weight of update gate, b_u is the bias of update gate and sigma is non-linear activation
# Result Gate:
# r[t] = sigma(X_eff[t].W_r[t] + b_r[t]), where r is the result gate output, W_r is the weight of result gate, b_r is the bias of result gate and sigma is non-linear activation
# Input:
# X'[t] = tanh(X_eff[t].W_c[t] + b_c[t]), here W_c and b_c are sizing parameters. Note that inside the cell, all the vectors computed lead to arrays of size n
# Whereas, the actual input is of size p and the effective input that we get after concatenation is of size p + n. This p + n size input needs to be mapped to n sized one.
# Now, the LSTM has a hidden state vector H and a memory state vector, denoted for some reason by C.
# Naturally, C should be updated as follows-- current memory should be previous memory scaled by forget gate answer added to the current input scaled by update gate answer. So,
# C[t] = f[t]*C[t - 1] + u[t]*X'[t], where * denotes element-wise product and not a matrix multiplication (which is being denoted by ' . ')
# And (not-so-)naturally, the hidden state should be updated (for some reason) as follows--
# H[t] = r[t]*tanh(C[t])
# A possibly plausible reasoning for this that r tells what proportion of the memory needs to be exposed to the outside as the answer and simultaneously remembered in state
# The output of the LSTM cell at time instant t is then given by the simple NN like equation--
# Y[t] = softmax(H[t]*W[t] + b[t]), where the W and b are weights and biases of the output gate.
# The obvious question here is that WHY WOULD ONE CHOOSE THESE EQUATIONS?? The choice of equations at all the gates and the internal computations seems to be arbitrary.
# The solution is GRU: Gated Recurrent Unit. It is a "cheaper" solution to LSTMs, as LSTMs have 3 necessar gates but GRU has only 2. (Okay, what is the big deal then?)
# The equations go as follows--
# X_eff[t] = X[t] | H[t - 1], has size p + n
# z[t] = sigma(X_eff[t].W_z[t] + b_z[t]), has size n
# r[t] = sigma(X_eff[t].W_r[t] + b_r[t]), has size n
# X'[t] = X[t] | r[t]*H[t - 1], has size p + n
# X''[t] = tanh(X'[t].W_c[t] + b_c[t]), has size n
# H[t] = (1 - z[t])*H[t - 1] + z[t]*X''[t], has size n
# Y[t] = softmax(H[t].W[t] + b[t]), has size m

