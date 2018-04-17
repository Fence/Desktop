import tensorflow as tf
from D_RNN import d_bi_RNN
import numpy as np
import random
from data_generator import NLPCC_data

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,"Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size.")
FLAGS = tf.app.flags.FLAGS
training_data=NLPCC_data()

model_path = "./models/model_13_02_6/"
rnn = d_bi_RNN(batch_size=FLAGS.batch_size,learning_rate=FLAGS.learning_rate,learning_rate_decay_factor=FLAGS.learning_rate_decay_factor)
# Initializing the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    dev_max_accuracy = 0.2
    dev_min_loss = 50
    model_saver = tf.train.Saver(max_to_keep=1000)
    # Keep training until reach max iterations
    while step * rnn.batch_size< rnn.training_iters:

        batch_x, batch_y,_ = training_data.next(rnn.batch_size)

        # dev_x = training_data.dev_data[:rnn.batch_size]        
        # dev_y = training_data.dev_labels[:rnn.batch_size]
        # dev_x = random.sample(training_data.dev_data,rnn.batch_size)       
        # dev_y = random.sample(training_data.dev_labels,rnn.batch_size)
        # Run optimization op (backprop)
        _, tr_acc, tr_loss ,pred,label= sess.run([rnn.optimizer, rnn.accuracy, rnn.cost,rnn.m,rnn.n],
                                feed_dict= {rnn.x_input: np.array(batch_x), rnn.y_input: np.array(batch_y),
                                rnn.rescaling: np.array([[22.33, 13.15,14.79, 9.40, 16.88, 0.49] for _ in range(len(batch_x))])})

        if step % rnn.display_step == 0:
            # Calculate batch accuracy and loss on dev set
            
            len_data = len(training_data.dev_data)
            correct_num =0
            loss=.0
            for batch in xrange(0, len_data, rnn.batch_size):
                new_batch = min(rnn.batch_size, len_data-batch)    
                rnn.batch_size= new_batch
                dev_x=training_data.dev_data[batch:batch+new_batch]
                dev_y=training_data.dev_labels[batch:batch+new_batch]
                dev_correct_num, dev_loss ,pred_dev,label_dev= sess.run([rnn.correct_num, rnn.cost,rnn.m,rnn.n],
                                                                feed_dict={rnn.x_input: np.array(dev_x), rnn.y_input: np.array(dev_y),
                                                                rnn.rescaling: np.array([[0.49,9.40,16.88,13.15,22.33,14.79] for _ in range(len(dev_x))])})
                correct_num+=dev_correct_num
                loss+=dev_loss*new_batch
                rnn.batch_size=FLAGS.batch_size
            # import ipdb;ipdb.set_trace()
            print(correct_num)
            print(len_data)
            dev_acc=float(correct_num)/len_data
            dev_avg_loss=loss/len_data
            print(" TRAIN DATA")
            print("Iter " + str(step*rnn.batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(tr_loss) + ", Training Accuracy= " +
                  "{:.5f}".format(tr_acc))
            print(pred)
            print(label)
            print(pred_dev)
            print(label_dev)
            print(" \nDEV DATA\n")

            if dev_acc>dev_max_accuracy:
                dev_max_accuracy = dev_acc
                checkpoint_path =model_path+"best_model.ckpt"
                save_path =  model_saver.save(sess= sess,save_path= checkpoint_path)
                print("Model saved in", save_path)

            if dev_loss<dev_min_loss:
                dev_min_loss = dev_avg_loss
            print("Loss", dev_avg_loss)
            print("accuracy", dev_acc)
            print("Min Loss", dev_min_loss)
            print("Max accuracy", dev_max_accuracy)
        if step % rnn.checkpoint_step == 0:
            checkpoint_path =model_path+str(step)+"_model.ckpt"
            save_path =  model_saver.save(sess= sess,save_path= checkpoint_path)
            sess.run(rnn.learning_rate_decay_op)
            print(sess.run(rnn.learning_rate),"haha")
            print("Model saved in", save_path)
        step += 1
    print("Optimization Finished!")
    sample_id=random.randint(0, 4000-rnn.batch_size)
    test_data = training_data.dev_data[sample_id:sample_id+rnn.batch_size]       
    test_label = training_data.dev_labels[sample_id:sample_id+rnn.batch_size]
    model_saver = tf.train.Saver()
    checkpoint_path =model_path+"best_model.ckpt"
    model_saver.restore(sess=sess, save_path=checkpoint_path)
    print ("Session Restored")

    print("Testing Accuracy:",
          sess.run(rnn.accuracy, feed_dict={rnn.x_input: test_data, rnn.y_input: test_label}))
