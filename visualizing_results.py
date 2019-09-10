import sys
import tensorboard
import tensorflow as tf
from tensorboard import main as tb
tf.flags.FLAGS.logdir = "--logdir /runs/1546559186/summaries"
tb.main()

print ("This is the name of the script: ", sys.argv[0])
print ("Number of arguments: ", len(sys.argv))
print ("The arguments are: " , str(sys.argv))

# sys.argv[1] = "--logdir"
# # --logdir /PATH_TO_CODE/runs/1449760558/summaries/
tensorboard(sys.argv)
