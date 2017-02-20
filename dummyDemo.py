import tensorflow as tf

x = tf.Variable(2, name="x")
y = tf.Variable(3, name="y")
f = x*x*x*y - y + 12

# This code does not actually perform any computation. 
# It just creates a computation graph. 
# The variables are not even initialized yet.

# To evaluate the graph create you need to open a TF session.
# Then you can initialize the variables and evaluate f. 

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# A session handles the operations onto devices and run it.
# It will hold all the variable values. 
# Close a session to frees up resources.