import math
import ipdb
import time
import numpy as np
import tensorflow as tf

from preprocessing import DataProcess


class ActFinder(object):
    """docstring for ActFinder"""
    def __init__(self, sess):
        self.sess = sess
        data_fold = -1
        databag = 'cooking'
        self.data_processer = DataProcess(data_fold, databag)
        self.buckets = self.data_processer.buckets
        self.vocab_size = len(self.data_processer.vocab)
        #self.vocab_size = args.vocab_size
        #self.word_dim = 128 #args.word_dim
        self.num_samples = 2048 #args.num_samples
        self.use_lstm = 1 #args.use_lstm
        self.cell_num = 128 #args.cell_num
        self.layer_num = 1 #args.layer_num
        #self.buckets = args.buckets
        self.max_gradient_norm = 5.0 #args.max_gradient_norm
        self.batch_size = 32 #batch_size
        self.decay = 0.99 #args.decay
        self.lr = 0.5 #args.init_lr
        self.build_model(self.buckets)


    def build_model(self, buckets, predict=False):
        self.learning_rate = tf.Variable(float(self.lr), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
                                self.learning_rate * self.decay)
        self.global_step = tf.Variable(0, trainable=False)
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if 0 < self.num_samples < self.vocab_size:
            w = tf.get_variable("proj_w", [self.cell_num, self.vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels,
                    self.num_samples, self.vocab_size)
            softmax_loss_function = sampled_loss

        a_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(self.cell_num)
        if self.use_lstm:
            a_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.cell_num)
        cell = a_cell
        if self.layer_num > 1:
            cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([a_cell] * self.layer_num)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_function(enc_in, dec_in, is_dec):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                enc_in, dec_in, cell, 
                num_encoder_symbols=self.vocab_size,
                num_decoder_symbols=self.vocab_size,
                embedding_size=self.cell_num,
                output_projection=output_projection,
                feed_previous=is_dec)
        # Feeds for inputs.
        self.enc_in = []
        self.dec_in = []
        self.target_w = []
        for i in xrange(buckets[-1][0]): # last bucket is the largest
            self.enc_in.append(tf.placeholder(tf.int32, shape=[None],
                                                name="enc_%d"%i))
        for i in xrange(buckets[-1][1] + 1):
            self.dec_in.append(tf.placeholder(tf.int32, shape=[None],
                                                name="dec_%d"%i))
            self.target_w.append(tf.placeholder(tf.float32, shape=[None],
                                                name="w_%d"%i))
        # Our targets are decoder inputs shifted by one.
        targets = [self.dec_in[i+1] for i in xrange(len(self.dec_in)-1)]

        if predict:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.enc_in, self.dec_in, targets, self.target_w, buckets,
                lambda x, y: seq2seq_function(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                            for output in self.outputs[b]
                    ]
        # Training outputs and losses.
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.enc_in, self.dec_in, targets, self.target_w, buckets,
                lambda x, y:seq2seq_function(x, y, False),
                softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not predict:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                    self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables())


    def step(self, enc_in, dec_in, target_w, bucket_id, predict):
        enc_size, dec_size = self.buckets[bucket_id]
        if len(enc_in) != enc_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(enc_in), enc_size))
        if len(dec_in) != dec_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(dec_in), dec_size))
        if len(target_w) != dec_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(target_w), dec_size))
        # Input feed: encoder inputs, decoder inputs, target_w, as provided.
        input_feed = {}
        for l in xrange(enc_size):
            input_feed[self.enc_in[l].name] = [enc_in[l]]
        for l in xrange(dec_size):
            input_feed[self.dec_in[l].name] = [dec_in[l]]
            input_feed[self.target_w[l].name] = [target_w[l]]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.dec_in[dec_size].name
        input_feed[last_target] = np.zeros(1) #np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not predict:
            output_feed = [self.updates[bucket_id], # Update Op that does SGD.
                            self.gradient_norms[bucket_id], # Gradient norm.
                            self.losses[bucket_id]] # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]] # Loss for this batch.
            for l in xrange(dec_size): # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = self.sess.run(output_feed, input_feed)
        if not predict:
            return outputs[1], outputs[2], None  # Gradient_norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient_norm, loss, outputs.


def train(outfile):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.24)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = ActFinder(sess)
        train_data, valid_data = model.data_processer.read_data() 
        epochs = 800

        train_bucket_sizes = [len(train_data[b]['sent']) for b in model.buckets]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print('train_buckets_scale: {}\n'.format(train_buckets_scale))
        outfile.write(('train_buckets_scale: {}\n'.format(train_buckets_scale)))
        # This is the training loop.
        loss = 0.0
        best_f1 = {'rec': 0.0, 'pre': 0.0, 'f1': 0.0}
        current_step = 0
        previous_losses = []
        for ep in xrange(epochs):
            #random_number_01 = np.random.random_sample()
            #bucket_id = min([i for i in xrange(len(train_buckets_scale))
            #               if train_buckets_scale[i] > random_number_01])
            bucket_id = ep % len(model.buckets)
            ep_steps = len(train_data[model.buckets[bucket_id]]['sent'])
            #print '\nep_steps: %d\n'%ep_steps
            for j in xrange(ep_steps):
                enc_in = train_data[model.buckets[bucket_id]]['sent'][j]
                dec_in = train_data[model.buckets[bucket_id]]['act'][j]
                t_w = train_data[model.buckets[bucket_id]]['t_w'][j]
                #N = len(enc_in)/model.batch_size

                _, step_loss, _ = model.step(enc_in, dec_in, t_w, bucket_id, False)
                loss += step_loss
                #if j%100 == 0:
                #    print 'ep: %d\tstep: %d\tloss: %f'%(ep, j, step_loss)
            #ipdb.set_trace()
            current_step += ep_steps
            loss /= ep_steps
            print('epoch = {}\tbucket = {}\tstep = {}\tloss = {}'.format(
                ep, bucket_id, current_step, loss))
            outfile.write('epoch = {}\tbucket = {}\tstep = {}\tloss = {}'.format(
                ep, bucket_id, current_step, loss))
            
            if (ep + 1) % 20 == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("\nglobal_step: %d\tlearning_rate: %.4f\tperplexity: %.2f" %
                   (model.global_step.eval(), model.learning_rate.eval(), perplexity))
                outfile.write("\nglobal_step: %d\tlearning_rate: %.4f\tperplexity: %.2f\n" %
                   (model.global_step.eval(), model.learning_rate.eval(), perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)

                    previous_losses.append(loss)

                right_act = 0
                total_act = 0
                tagged_act = 0
                precision = 0
                f_measure = 0
                recall = 0
                #ipdb.set_trace()
                for bucket_id in xrange(len(model.buckets)):
                    for i in xrange(len(valid_data[model.buckets[bucket_id]]['sent'])):
                        enc_in = valid_data[model.buckets[bucket_id]]['sent'][i]
                        dec_in = valid_data[model.buckets[bucket_id]]['act'][i]
                        t_w = valid_data[model.buckets[bucket_id]]['t_w'][i]

                        _, eval_loss, out_logits = model.step(enc_in, dec_in, t_w, bucket_id, True)

                        predicted_actions = []
                        for logit in out_logits:
                            selected_token_id = int(np.argmax(logit, axis=1))
                            if selected_token_id == 0: # EOF id
                                break
                            else:
                                if selected_token_id not in predicted_actions:
                                    predicted_actions.append(selected_token_id)
                        total_act += len([k for k in dec_in if k != 0])
                        tagged_act += len(predicted_actions)
                        for a in predicted_actions:
                            if a in dec_in:
                                right_act += 1
                        #eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        #print("  eval: bucket {}  perplexity {}".format(b, eval_ppx))
                if total_act > 0:
                    if tagged_act > 0:
                        precision = float(right_act) / tagged_act
                        recall = float(right_act) / total_act
                    if precision + recall > 0:
                        f_measure = 2 * precision * recall / (precision + recall)
                if best_f1['f1'] < f_measure:
                    best_f1['f1'] = f_measure
                    best_f1['rec'] = recall
                    best_f1['pre'] = precision
                print('total_act:%d\ntagged_act:%d\nright_act:%d'%(total_act, tagged_act, right_act))
                print('recall:%f\nprecision:%f\nf_measure:%f'%(recall, precision, f_measure))
                outfile.write('total_act:%d\ntagged_act:%d\nright_act:%d\n'%(total_act, tagged_act, right_act))
                outfile.write('recall:%f\nprecision:%f\nf_measure:%f\n'%(recall, precision, f_measure))
                outfile.write('best_f1: {}\n'.format(best_f1))

if __name__ == '__main__':
    #ipdb.set_trace()
    start = time.time()
    with open('test_result1.txt', 'w') as outfile:
        train(outfile)
        end = time.time()
        print 'total time cost: %ds' % (end - start)
        outfile.write('total time cost: %ds' % (end - start))
