nohup: ignoring input
Namespace(batch_size=64, brnn=True, brnn_merge='concat', curriculum=False, data='data-bpe/len80.train.pt', dropout=0.3, epochs=30, extra_shuffle=False, gpus=[1], input_feed=1, layers=2, learning_rate=1.0, learning_rate_decay=0.5, log_interval=50, max_generator_batches=32, max_grad_norm=5, optim='sgd', param_init=0.1, pre_word_vecs_dec=None, pre_word_vecs_enc=None, rnn_size=500, save_model='model.bi', start_decay_at=8, start_epoch=1, train_from='', train_from_state_dict='', word_vec_size=500)
Loading data from 'data-bpe/len80.train.pt'
 * vocabulary size. source = 15114; target = 9916
 * number of training sentences. 96046
 * maximum batch size. 64
Building model...
* number of parameters: 26248916
NMTModel (
  (encoder): Encoder (
    (word_lut): Embedding(15114, 500, padding_idx=0)
    (rnn): LSTM(500, 250, num_layers=2, dropout=0.3, bidirectional=True)
  )
  (decoder): Decoder (
    (word_lut): Embedding(9916, 500, padding_idx=0)
    (rnn): StackedLSTM (
      (dropout): Dropout (p = 0.3)
      (layers): ModuleList (
        (0): LSTMCell(1000, 500)
        (1): LSTMCell(500, 500)
      )
    )
    (attn): GlobalAttention (
      (linear_in): Linear (500 -> 500)
      (sm): Softmax ()
      (linear_out): Linear (1000 -> 500)
      (tanh): Tanh ()
    )
    (dropout): Dropout (p = 0.3)
  )
  (generator): Sequential (
    (0): Linear (500 -> 9916)
    (1): LogSoftmax ()
  )
)

Epoch  1,    50/ 1501; acc:   0.00; ppl: 16226.82; 3223 src tok/s; 4122 tgt tok/s;     30 s elapsed
Epoch  1,   100/ 1501; acc:   0.00; ppl: 6742.18; 3308 src tok/s; 4237 tgt tok/s;     60 s elapsed
Epoch  1,   150/ 1501; acc:   0.00; ppl: 5508.81; 3068 src tok/s; 4043 tgt tok/s;     86 s elapsed
Epoch  1,   200/ 1501; acc:   0.00; ppl: 3027.99; 3205 src tok/s; 4166 tgt tok/s;    113 s elapsed
Epoch  1,   250/ 1501; acc:   0.00; ppl: 1439.90; 2880 src tok/s; 3788 tgt tok/s;    141 s elapsed
Epoch  1,   300/ 1501; acc:   0.00; ppl: 1059.72; 2891 src tok/s; 3778 tgt tok/s;    171 s elapsed
Epoch  1,   350/ 1501; acc:   0.00; ppl: 881.47; 3363 src tok/s; 4302 tgt tok/s;    200 s elapsed
Epoch  1,   400/ 1501; acc:   0.00; ppl: 714.89; 3094 src tok/s; 3995 tgt tok/s;    229 s elapsed
Epoch  1,   450/ 1501; acc:   0.00; ppl: 630.63; 3235 src tok/s; 4263 tgt tok/s;    254 s elapsed
Epoch  1,   500/ 1501; acc:   0.00; ppl: 552.14; 3291 src tok/s; 4226 tgt tok/s;    281 s elapsed
Epoch  1,   550/ 1501; acc:   0.00; ppl: 486.87; 3225 src tok/s; 4199 tgt tok/s;    308 s elapsed
Epoch  1,   600/ 1501; acc:   0.00; ppl: 453.18; 3206 src tok/s; 4172 tgt tok/s;    336 s elapsed
Epoch  1,   650/ 1501; acc:   0.00; ppl: 417.57; 3224 src tok/s; 4166 tgt tok/s;    366 s elapsed
Epoch  1,   700/ 1501; acc:   0.00; ppl: 377.00; 3188 src tok/s; 4119 tgt tok/s;    396 s elapsed
Epoch  1,   750/ 1501; acc:   0.00; ppl: 353.69; 3305 src tok/s; 4247 tgt tok/s;    423 s elapsed
Epoch  1,   800/ 1501; acc:   0.00; ppl: 313.12; 3191 src tok/s; 4165 tgt tok/s;    451 s elapsed
Epoch  1,   850/ 1501; acc:   0.00; ppl: 302.37; 3310 src tok/s; 4246 tgt tok/s;    479 s elapsed
Epoch  1,   900/ 1501; acc:   0.00; ppl: 285.39; 3340 src tok/s; 4262 tgt tok/s;    507 s elapsed
Epoch  1,   950/ 1501; acc:   0.00; ppl: 268.72; 3379 src tok/s; 4323 tgt tok/s;    536 s elapsed
Epoch  1,  1000/ 1501; acc:   0.00; ppl: 241.88; 3227 src tok/s; 4184 tgt tok/s;    565 s elapsed
Epoch  1,  1050/ 1501; acc:   0.00; ppl: 226.89; 2951 src tok/s; 3889 tgt tok/s;    592 s elapsed
Epoch  1,  1100/ 1501; acc:   0.00; ppl: 216.50; 3231 src tok/s; 4188 tgt tok/s;    618 s elapsed
Epoch  1,  1150/ 1501; acc:   0.00; ppl: 211.67; 3301 src tok/s; 4278 tgt tok/s;    644 s elapsed
Epoch  1,  1200/ 1501; acc:   0.00; ppl: 203.10; 3125 src tok/s; 4080 tgt tok/s;    671 s elapsed
Epoch  1,  1250/ 1501; acc:   0.00; ppl: 192.47; 3259 src tok/s; 4164 tgt tok/s;    699 s elapsed
Epoch  1,  1300/ 1501; acc:   0.00; ppl: 186.53; 3205 src tok/s; 4172 tgt tok/s;    729 s elapsed
Epoch  1,  1350/ 1501; acc:   0.00; ppl: 180.51; 3269 src tok/s; 4226 tgt tok/s;    756 s elapsed
Epoch  1,  1400/ 1501; acc:   0.00; ppl: 171.61; 3037 src tok/s; 4019 tgt tok/s;    783 s elapsed
Epoch  1,  1450/ 1501; acc:   0.00; ppl: 169.08; 3278 src tok/s; 4215 tgt tok/s;    811 s elapsed
Epoch  1,  1500/ 1501; acc:   0.00; ppl: 162.19; 3312 src tok/s; 4251 tgt tok/s;    840 s elapsed
Train perplexity: 499.977
Train accuracy: 0
Validation perplexity: 230.289
Validation accuracy: 0

Epoch  2,    50/ 1501; acc:   0.00; ppl: 150.99; 3316 src tok/s; 4251 tgt tok/s;    870 s elapsed
Epoch  2,   100/ 1501; acc:   0.00; ppl: 149.27; 3219 src tok/s; 4181 tgt tok/s;    898 s elapsed
Epoch  2,   150/ 1501; acc:   0.00; ppl: 142.05; 3207 src tok/s; 4158 tgt tok/s;    926 s elapsed
Epoch  2,   200/ 1501; acc:   0.00; ppl: 145.03; 3366 src tok/s; 4258 tgt tok/s;    955 s elapsed
Epoch  2,   250/ 1501; acc:   0.00; ppl: 139.26; 3188 src tok/s; 4139 tgt tok/s;    982 s elapsed
Epoch  2,   300/ 1501; acc:   0.00; ppl: 136.78; 3235 src tok/s; 4214 tgt tok/s;   1010 s elapsed
Epoch  2,   350/ 1501; acc:   0.00; ppl: 131.15; 3112 src tok/s; 4001 tgt tok/s;   1037 s elapsed
Epoch  2,   400/ 1501; acc:   0.00; ppl: 132.38; 3256 src tok/s; 4147 tgt tok/s;   1066 s elapsed
Epoch  2,   450/ 1501; acc:   0.00; ppl: 123.85; 3159 src tok/s; 4132 tgt tok/s;   1094 s elapsed
Epoch  2,   500/ 1501; acc:   0.00; ppl: 123.94; 3193 src tok/s; 4153 tgt tok/s;   1121 s elapsed
Epoch  2,   550/ 1501; acc:   0.00; ppl: 123.86; 3184 src tok/s; 4088 tgt tok/s;   1153 s elapsed
Epoch  2,   600/ 1501; acc:   0.00; ppl: 114.09; 3041 src tok/s; 3929 tgt tok/s;   1181 s elapsed
Epoch  2,   650/ 1501; acc:   0.00; ppl: 110.65; 3012 src tok/s; 3963 tgt tok/s;   1209 s elapsed
Epoch  2,   700/ 1501; acc:   0.00; ppl: 106.87; 3136 src tok/s; 4076 tgt tok/s;   1239 s elapsed
Epoch  2,   750/ 1501; acc:   0.00; ppl: 105.69; 3146 src tok/s; 4112 tgt tok/s;   1265 s elapsed
Epoch  2,   800/ 1501; acc:   0.00; ppl: 101.36; 3076 src tok/s; 4039 tgt tok/s;   1292 s elapsed
Epoch  2,   850/ 1501; acc:   0.00; ppl: 106.17; 3165 src tok/s; 4119 tgt tok/s;   1319 s elapsed
Epoch  2,   900/ 1501; acc:   0.00; ppl: 101.89; 3210 src tok/s; 4178 tgt tok/s;   1347 s elapsed
Epoch  2,   950/ 1501; acc:   0.00; ppl: 104.26; 3262 src tok/s; 4176 tgt tok/s;   1377 s elapsed
Epoch  2,  1000/ 1501; acc:   0.00; ppl:  98.71; 3062 src tok/s; 3991 tgt tok/s;   1405 s elapsed
Epoch  2,  1050/ 1501; acc:   0.00; ppl:  92.24; 3132 src tok/s; 4056 tgt tok/s;   1432 s elapsed
Epoch  2,  1100/ 1501; acc:   0.00; ppl:  96.52; 3349 src tok/s; 4258 tgt tok/s;   1463 s elapsed
Epoch  2,  1150/ 1501; acc:   0.00; ppl:  91.21; 3186 src tok/s; 4108 tgt tok/s;   1491 s elapsed
Epoch  2,  1200/ 1501; acc:   0.00; ppl:  86.16; 3126 src tok/s; 4113 tgt tok/s;   1517 s elapsed
Epoch  2,  1250/ 1501; acc:   0.00; ppl:  85.15; 3231 src tok/s; 4167 tgt tok/s;   1545 s elapsed
Epoch  2,  1300/ 1501; acc:   0.00; ppl:  87.14; 3526 src tok/s; 4508 tgt tok/s;   1574 s elapsed
Epoch  2,  1350/ 1501; acc:   0.00; ppl:  81.83; 3294 src tok/s; 4304 tgt tok/s;   1598 s elapsed
Epoch  2,  1400/ 1501; acc:   0.00; ppl:  84.59; 3310 src tok/s; 4303 tgt tok/s;   1625 s elapsed
Epoch  2,  1450/ 1501; acc:   0.00; ppl:  81.76; 3364 src tok/s; 4358 tgt tok/s;   1654 s elapsed
Epoch  2,  1500/ 1501; acc:   0.00; ppl:  76.02; 3259 src tok/s; 4276 tgt tok/s;   1679 s elapsed
Train perplexity: 108.29
Train accuracy: 0
Validation perplexity: 116.647
Validation accuracy: 0

Epoch  3,    50/ 1501; acc:   0.00; ppl:  67.86; 3101 src tok/s; 4124 tgt tok/s;   1706 s elapsed
Epoch  3,   100/ 1501; acc:   0.00; ppl:  74.25; 3448 src tok/s; 4440 tgt tok/s;   1732 s elapsed
Epoch  3,   150/ 1501; acc:   0.00; ppl:  69.14; 3333 src tok/s; 4338 tgt tok/s;   1759 s elapsed
Epoch  3,   200/ 1501; acc:   0.00; ppl:  68.14; 3300 src tok/s; 4241 tgt tok/s;   1786 s elapsed
Epoch  3,   250/ 1501; acc:   0.00; ppl:  63.33; 3228 src tok/s; 4240 tgt tok/s;   1812 s elapsed
Epoch  3,   300/ 1501; acc:   0.00; ppl:  68.63; 3483 src tok/s; 4455 tgt tok/s;   1839 s elapsed
Epoch  3,   350/ 1501; acc:   0.00; ppl:  64.81; 3257 src tok/s; 4220 tgt tok/s;   1866 s elapsed
Epoch  3,   400/ 1501; acc:   0.00; ppl:  63.39; 3338 src tok/s; 4306 tgt tok/s;   1895 s elapsed
Epoch  3,   450/ 1501; acc:   0.00; ppl:  63.68; 3299 src tok/s; 4261 tgt tok/s;   1923 s elapsed
Epoch  3,   500/ 1501; acc:   0.00; ppl:  60.48; 2920 src tok/s; 3814 tgt tok/s;   1952 s elapsed
Epoch  3,   550/ 1501; acc:   0.00; ppl:  60.96; 3001 src tok/s; 3892 tgt tok/s;   1984 s elapsed
Epoch  3,   600/ 1501; acc:   0.00; ppl:  56.26; 2941 src tok/s; 3834 tgt tok/s;   2013 s elapsed
Epoch  3,   650/ 1501; acc:   0.00; ppl:  54.26; 3111 src tok/s; 4074 tgt tok/s;   2041 s elapsed
Epoch  3,   700/ 1501; acc:   0.00; ppl:  55.37; 3199 src tok/s; 4175 tgt tok/s;   2068 s elapsed
Epoch  3,   750/ 1501; acc:   0.00; ppl:  54.99; 3093 src tok/s; 4022 tgt tok/s;   2096 s elapsed
Epoch  3,   800/ 1501; acc:   0.00; ppl:  55.80; 3131 src tok/s; 4068 tgt tok/s;   2126 s elapsed
Epoch  3,   850/ 1501; acc:   0.00; ppl:  55.36; 3254 src tok/s; 4170 tgt tok/s;   2154 s elapsed
Epoch  3,   900/ 1501; acc:   0.00; ppl:  53.30; 3206 src tok/s; 4095 tgt tok/s;   2182 s elapsed
Epoch  3,   950/ 1501; acc:   0.00; ppl:  49.01; 3083 src tok/s; 4091 tgt tok/s;   2207 s elapsed
Epoch  3,  1000/ 1501; acc:   0.00; ppl:  52.34; 3222 src tok/s; 4181 tgt tok/s;   2237 s elapsed
Epoch  3,  1050/ 1501; acc:   0.00; ppl:  49.82; 3103 src tok/s; 4053 tgt tok/s;   2263 s elapsed
Epoch  3,  1100/ 1501; acc:   0.00; ppl:  52.98; 3325 src tok/s; 4208 tgt tok/s;   2292 s elapsed
Epoch  3,  1150/ 1501; acc:   0.00; ppl:  47.46; 3243 src tok/s; 4213 tgt tok/s;   2321 s elapsed
Epoch  3,  1200/ 1501; acc:   0.00; ppl:  48.39; 3113 src tok/s; 4050 tgt tok/s;   2347 s elapsed
Epoch  3,  1250/ 1501; acc:   0.00; ppl:  48.32; 3244 src tok/s; 4187 tgt tok/s;   2376 s elapsed
Epoch  3,  1300/ 1501; acc:   0.00; ppl:  45.46; 3321 src tok/s; 4240 tgt tok/s;   2403 s elapsed
Epoch  3,  1350/ 1501; acc:   0.00; ppl:  48.84; 3267 src tok/s; 4160 tgt tok/s;   2433 s elapsed
Epoch  3,  1400/ 1501; acc:   0.00; ppl:  46.21; 3231 src tok/s; 4158 tgt tok/s;   2462 s elapsed
Epoch  3,  1450/ 1501; acc:   0.00; ppl:  43.42; 3173 src tok/s; 4122 tgt tok/s;   2490 s elapsed
Epoch  3,  1500/ 1501; acc:   0.00; ppl:  45.10; 3362 src tok/s; 4322 tgt tok/s;   2520 s elapsed
Train perplexity: 55.5712
Train accuracy: 0
Validation perplexity: 63.7461
Validation accuracy: 0

Epoch  4,    50/ 1501; acc:   0.00; ppl:  42.61; 3316 src tok/s; 4231 tgt tok/s;   2549 s elapsed
Epoch  4,   100/ 1501; acc:   0.00; ppl:  39.42; 3304 src tok/s; 4273 tgt tok/s;   2577 s elapsed
Epoch  4,   150/ 1501; acc:   0.00; ppl:  38.21; 3135 src tok/s; 4121 tgt tok/s;   2603 s elapsed
Epoch  4,   200/ 1501; acc:   0.00; ppl:  36.48; 3092 src tok/s; 4066 tgt tok/s;   2630 s elapsed
Epoch  4,   250/ 1501; acc:   0.00; ppl:  35.14; 3043 src tok/s; 4007 tgt tok/s;   2656 s elapsed
Epoch  4,   300/ 1501; acc:   0.00; ppl:  38.07; 3310 src tok/s; 4268 tgt tok/s;   2685 s elapsed
Epoch  4,   350/ 1501; acc:   0.00; ppl:  38.86; 3162 src tok/s; 4065 tgt tok/s;   2713 s elapsed
Epoch  4,   400/ 1501; acc:   0.00; ppl:  39.11; 3401 src tok/s; 4316 tgt tok/s;   2743 s elapsed
Epoch  4,   450/ 1501; acc:   0.00; ppl:  37.03; 3278 src tok/s; 4275 tgt tok/s;   2772 s elapsed
Epoch  4,   500/ 1501; acc:   0.00; ppl:  36.61; 3166 src tok/s; 4126 tgt tok/s;   2798 s elapsed
Epoch  4,   550/ 1501; acc:   0.00; ppl:  37.88; 3381 src tok/s; 4314 tgt tok/s;   2827 s elapsed
Epoch  4,   600/ 1501; acc:   0.00; ppl:  37.39; 3258 src tok/s; 4164 tgt tok/s;   2855 s elapsed
Epoch  4,   650/ 1501; acc:   0.00; ppl:  35.17; 3189 src tok/s; 4167 tgt tok/s;   2883 s elapsed
Epoch  4,   700/ 1501; acc:   0.00; ppl:  35.50; 3257 src tok/s; 4204 tgt tok/s;   2911 s elapsed
Epoch  4,   750/ 1501; acc:   0.00; ppl:  34.97; 3265 src tok/s; 4214 tgt tok/s;   2940 s elapsed
Epoch  4,   800/ 1501; acc:   0.00; ppl:  36.12; 3426 src tok/s; 4377 tgt tok/s;   2970 s elapsed
Epoch  4,   850/ 1501; acc:   0.00; ppl:  31.96; 3129 src tok/s; 4138 tgt tok/s;   2995 s elapsed
Epoch  4,   900/ 1501; acc:   0.00; ppl:  34.56; 2941 src tok/s; 3839 tgt tok/s;   3026 s elapsed
Epoch  4,   950/ 1501; acc:   0.00; ppl:  33.90; 3207 src tok/s; 4092 tgt tok/s;   3054 s elapsed
Epoch  4,  1000/ 1501; acc:   0.00; ppl:  31.64; 3008 src tok/s; 3928 tgt tok/s;   3082 s elapsed
Epoch  4,  1050/ 1501; acc:   0.00; ppl:  35.20; 3247 src tok/s; 4123 tgt tok/s;   3110 s elapsed
Epoch  4,  1100/ 1501; acc:   0.00; ppl:  34.43; 3314 src tok/s; 4269 tgt tok/s;   3139 s elapsed
Epoch  4,  1150/ 1501; acc:   0.00; ppl:  31.84; 3177 src tok/s; 4175 tgt tok/s;   3164 s elapsed
Epoch  4,  1200/ 1501; acc:   0.00; ppl:  34.11; 3111 src tok/s; 4065 tgt tok/s;   3192 s elapsed
Epoch  4,  1250/ 1501; acc:   0.00; ppl:  32.17; 3008 src tok/s; 3896 tgt tok/s;   3221 s elapsed
Epoch  4,  1300/ 1501; acc:   0.00; ppl:  33.55; 3104 src tok/s; 4004 tgt tok/s;   3253 s elapsed
Epoch  4,  1350/ 1501; acc:   0.00; ppl:  31.64; 3137 src tok/s; 4071 tgt tok/s;   3281 s elapsed
Epoch  4,  1400/ 1501; acc:   0.00; ppl:  32.07; 3134 src tok/s; 4073 tgt tok/s;   3310 s elapsed
Epoch  4,  1450/ 1501; acc:   0.00; ppl:  29.16; 3045 src tok/s; 4017 tgt tok/s;   3337 s elapsed
Epoch  4,  1500/ 1501; acc:   0.00; ppl:  31.11; 3170 src tok/s; 4092 tgt tok/s;   3365 s elapsed
Train perplexity: 35.1374
Train accuracy: 0
Validation perplexity: 44.2513
Validation accuracy: 0

Epoch  5,    50/ 1501; acc:   0.00; ppl:  27.79; 3187 src tok/s; 4150 tgt tok/s;   3396 s elapsed
Epoch  5,   100/ 1501; acc:   0.00; ppl:  27.03; 3207 src tok/s; 4160 tgt tok/s;   3423 s elapsed
Epoch  5,   150/ 1501; acc:   0.00; ppl:  27.08; 3259 src tok/s; 4217 tgt tok/s;   3451 s elapsed
Epoch  5,   200/ 1501; acc:   0.00; ppl:  29.22; 3220 src tok/s; 4140 tgt tok/s;   3479 s elapsed
Epoch  5,   250/ 1501; acc:   0.00; ppl:  27.18; 3281 src tok/s; 4245 tgt tok/s;   3507 s elapsed
Epoch  5,   300/ 1501; acc:   0.00; ppl:  27.48; 3238 src tok/s; 4216 tgt tok/s;   3533 s elapsed
Epoch  5,   350/ 1501; acc:   0.00; ppl:  27.23; 3183 src tok/s; 4166 tgt tok/s;   3561 s elapsed
Epoch  5,   400/ 1501; acc:   0.00; ppl:  29.99; 3392 src tok/s; 4250 tgt tok/s;   3590 s elapsed
Epoch  5,   450/ 1501; acc:   0.00; ppl:  26.75; 3230 src tok/s; 4216 tgt tok/s;   3618 s elapsed
Epoch  5,   500/ 1501; acc:   0.00; ppl:  27.09; 3234 src tok/s; 4159 tgt tok/s;   3645 s elapsed
Epoch  5,   550/ 1501; acc:   0.00; ppl:  28.71; 3382 src tok/s; 4270 tgt tok/s;   3675 s elapsed
Epoch  5,   600/ 1501; acc:   0.00; ppl:  27.46; 3081 src tok/s; 4026 tgt tok/s;   3703 s elapsed
Epoch  5,   650/ 1501; acc:   0.00; ppl:  26.35; 3291 src tok/s; 4280 tgt tok/s;   3732 s elapsed
Epoch  5,   700/ 1501; acc:   0.00; ppl:  26.94; 3313 src tok/s; 4287 tgt tok/s;   3758 s elapsed
Epoch  5,   750/ 1501; acc:   0.00; ppl:  25.67; 3086 src tok/s; 4062 tgt tok/s;   3786 s elapsed
Epoch  5,   800/ 1501; acc:   0.00; ppl:  27.86; 3342 src tok/s; 4287 tgt tok/s;   3814 s elapsed
Epoch  5,   850/ 1501; acc:   0.00; ppl:  26.51; 3117 src tok/s; 4043 tgt tok/s;   3844 s elapsed
Epoch  5,   900/ 1501; acc:   0.00; ppl:  25.78; 3212 src tok/s; 4153 tgt tok/s;   3870 s elapsed
Epoch  5,   950/ 1501; acc:   0.00; ppl:  24.46; 3037 src tok/s; 4033 tgt tok/s;   3896 s elapsed
Epoch  5,  1000/ 1501; acc:   0.00; ppl:  26.18; 3095 src tok/s; 4038 tgt tok/s;   3924 s elapsed
Epoch  5,  1050/ 1501; acc:   0.00; ppl:  26.15; 3125 src tok/s; 4018 tgt tok/s;   3952 s elapsed
Epoch  5,  1100/ 1501; acc:   0.00; ppl:  26.17; 3135 src tok/s; 4078 tgt tok/s;   3979 s elapsed
Epoch  5,  1150/ 1501; acc:   0.00; ppl:  25.91; 3238 src tok/s; 4181 tgt tok/s;   4007 s elapsed
Epoch  5,  1200/ 1501; acc:   0.00; ppl:  27.14; 3285 src tok/s; 4154 tgt tok/s;   4037 s elapsed
Epoch  5,  1250/ 1501; acc:   0.00; ppl:  25.72; 3152 src tok/s; 4085 tgt tok/s;   4065 s elapsed
Epoch  5,  1300/ 1501; acc:   0.00; ppl:  24.07; 3131 src tok/s; 4122 tgt tok/s;   4092 s elapsed
Epoch  5,  1350/ 1501; acc:   0.00; ppl:  25.05; 3132 src tok/s; 4070 tgt tok/s;   4122 s elapsed
Epoch  5,  1400/ 1501; acc:   0.00; ppl:  24.54; 3173 src tok/s; 4141 tgt tok/s;   4148 s elapsed
Epoch  5,  1450/ 1501; acc:   0.00; ppl:  26.47; 3269 src tok/s; 4225 tgt tok/s;   4177 s elapsed
Epoch  5,  1500/ 1501; acc:   0.00; ppl:  25.02; 3284 src tok/s; 4267 tgt tok/s;   4204 s elapsed
Train perplexity: 26.6411
Train accuracy: 0
Validation perplexity: 34.754
Validation accuracy: 0

Epoch  6,    50/ 1501; acc:   0.00; ppl:  21.94; 3288 src tok/s; 4242 tgt tok/s;   4235 s elapsed
Epoch  6,   100/ 1501; acc:   0.00; ppl:  22.55; 3106 src tok/s; 4035 tgt tok/s;   4262 s elapsed
Epoch  6,   150/ 1501; acc:   0.00; ppl:  21.64; 3092 src tok/s; 4045 tgt tok/s;   4290 s elapsed
Epoch  6,   200/ 1501; acc:   0.00; ppl:  23.54; 3250 src tok/s; 4187 tgt tok/s;   4320 s elapsed
Epoch  6,   250/ 1501; acc:   0.00; ppl:  21.74; 3274 src tok/s; 4248 tgt tok/s;   4346 s elapsed
Epoch  6,   300/ 1501; acc:   0.00; ppl:  23.68; 3324 src tok/s; 4257 tgt tok/s;   4376 s elapsed
Epoch  6,   350/ 1501; acc:   0.00; ppl:  22.18; 3122 src tok/s; 4048 tgt tok/s;   4404 s elapsed
Epoch  6,   400/ 1501; acc:   0.00; ppl:  21.98; 3260 src tok/s; 4235 tgt tok/s;   4433 s elapsed
Epoch  6,   450/ 1501; acc:   0.00; ppl:  22.33; 3206 src tok/s; 4197 tgt tok/s;   4462 s elapsed
Epoch  6,   500/ 1501; acc:   0.00; ppl:  20.70; 3165 src tok/s; 4132 tgt tok/s;   4489 s elapsed
Epoch  6,   550/ 1501; acc:   0.00; ppl:  21.02; 3144 src tok/s; 4123 tgt tok/s;   4515 s elapsed
Epoch  6,   600/ 1501; acc:   0.00; ppl:  21.11; 3092 src tok/s; 4060 tgt tok/s;   4542 s elapsed
Epoch  6,   650/ 1501; acc:   0.00; ppl:  22.63; 3272 src tok/s; 4240 tgt tok/s;   4570 s elapsed
Epoch  6,   700/ 1501; acc:   0.00; ppl:  23.98; 3349 src tok/s; 4266 tgt tok/s;   4599 s elapsed
Epoch  6,   750/ 1501; acc:   0.00; ppl:  21.68; 3236 src tok/s; 4176 tgt tok/s;   4628 s elapsed
Epoch  6,   800/ 1501; acc:   0.00; ppl:  22.04; 3154 src tok/s; 4024 tgt tok/s;   4656 s elapsed
Epoch  6,   850/ 1501; acc:   0.00; ppl:  22.93; 3331 src tok/s; 4231 tgt tok/s;   4684 s elapsed
Epoch  6,   900/ 1501; acc:   0.00; ppl:  21.14; 3164 src tok/s; 4144 tgt tok/s;   4710 s elapsed
Epoch  6,   950/ 1501; acc:   0.00; ppl:  22.61; 3159 src tok/s; 4058 tgt tok/s;   4739 s elapsed
Epoch  6,  1000/ 1501; acc:   0.00; ppl:  23.38; 3083 src tok/s; 3947 tgt tok/s;   4770 s elapsed
Epoch  6,  1050/ 1501; acc:   0.00; ppl:  21.78; 3189 src tok/s; 4109 tgt tok/s;   4797 s elapsed
Epoch  6,  1100/ 1501; acc:   0.00; ppl:  20.77; 2981 src tok/s; 3916 tgt tok/s;   4826 s elapsed
Epoch  6,  1150/ 1501; acc:   0.00; ppl:  22.99; 3223 src tok/s; 4121 tgt tok/s;   4854 s elapsed
Epoch  6,  1200/ 1501; acc:   0.00; ppl:  22.04; 3277 src tok/s; 4199 tgt tok/s;   4881 s elapsed
Epoch  6,  1250/ 1501; acc:   0.00; ppl:  19.87; 3011 src tok/s; 4017 tgt tok/s;   4907 s elapsed
Epoch  6,  1300/ 1501; acc:   0.00; ppl:  21.52; 3215 src tok/s; 4160 tgt tok/s;   4936 s elapsed
Epoch  6,  1350/ 1501; acc:   0.00; ppl:  21.40; 3171 src tok/s; 4070 tgt tok/s;   4963 s elapsed
Epoch  6,  1400/ 1501; acc:   0.00; ppl:  21.24; 3132 src tok/s; 4066 tgt tok/s;   4993 s elapsed
Epoch  6,  1450/ 1501; acc:   0.00; ppl:  21.45; 3301 src tok/s; 4287 tgt tok/s;   5023 s elapsed
Epoch  6,  1500/ 1501; acc:   0.00; ppl:  21.81; 3191 src tok/s; 4167 tgt tok/s;   5049 s elapsed
Train perplexity: 21.9981
Train accuracy: 0
Validation perplexity: 29.3038
Validation accuracy: 0

Epoch  7,    50/ 1501; acc:   0.00; ppl:  19.12; 3159 src tok/s; 4099 tgt tok/s;   5081 s elapsed
Epoch  7,   100/ 1501; acc:   0.00; ppl:  18.15; 3104 src tok/s; 4092 tgt tok/s;   5108 s elapsed
Epoch  7,   150/ 1501; acc:   0.00; ppl:  17.41; 2979 src tok/s; 3936 tgt tok/s;   5137 s elapsed
Epoch  7,   200/ 1501; acc:   0.00; ppl:  18.45; 3217 src tok/s; 4123 tgt tok/s;   5165 s elapsed
Epoch  7,   250/ 1501; acc:   0.00; ppl:  18.30; 3177 src tok/s; 4160 tgt tok/s;   5192 s elapsed
Epoch  7,   300/ 1501; acc:   0.00; ppl:  19.07; 3216 src tok/s; 4154 tgt tok/s;   5219 s elapsed
Epoch  7,   350/ 1501; acc:   0.00; ppl:  18.67; 3116 src tok/s; 4066 tgt tok/s;   5248 s elapsed
Epoch  7,   400/ 1501; acc:   0.00; ppl:  19.41; 3239 src tok/s; 4180 tgt tok/s;   5275 s elapsed
Epoch  7,   450/ 1501; acc:   0.00; ppl:  20.44; 3276 src tok/s; 4145 tgt tok/s;   5307 s elapsed
Epoch  7,   500/ 1501; acc:   0.00; ppl:  18.40; 3219 src tok/s; 4243 tgt tok/s;   5332 s elapsed
Epoch  7,   550/ 1501; acc:   0.00; ppl:  20.62; 3201 src tok/s; 4101 tgt tok/s;   5362 s elapsed
Epoch  7,   600/ 1501; acc:   0.00; ppl:  18.82; 3286 src tok/s; 4295 tgt tok/s;   5388 s elapsed
Epoch  7,   650/ 1501; acc:   0.00; ppl:  19.10; 3078 src tok/s; 3989 tgt tok/s;   5416 s elapsed
Epoch  7,   700/ 1501; acc:   0.00; ppl:  19.00; 3177 src tok/s; 4111 tgt tok/s;   5444 s elapsed
Epoch  7,   750/ 1501; acc:   0.00; ppl:  18.89; 3199 src tok/s; 4127 tgt tok/s;   5472 s elapsed
Epoch  7,   800/ 1501; acc:   0.00; ppl:  19.99; 3158 src tok/s; 4075 tgt tok/s;   5501 s elapsed
Epoch  7,   850/ 1501; acc:   0.00; ppl:  20.24; 3253 src tok/s; 4159 tgt tok/s;   5531 s elapsed
Epoch  7,   900/ 1501; acc:   0.00; ppl:  20.28; 3326 src tok/s; 4248 tgt tok/s;   5560 s elapsed
Epoch  7,   950/ 1501; acc:   0.00; ppl:  19.08; 3136 src tok/s; 4094 tgt tok/s;   5590 s elapsed
Epoch  7,  1000/ 1501; acc:   0.00; ppl:  18.26; 3132 src tok/s; 4095 tgt tok/s;   5617 s elapsed
Epoch  7,  1050/ 1501; acc:   0.00; ppl:  19.88; 3117 src tok/s; 4035 tgt tok/s;   5646 s elapsed
Epoch  7,  1100/ 1501; acc:   0.00; ppl:  19.13; 3108 src tok/s; 4049 tgt tok/s;   5673 s elapsed
Epoch  7,  1150/ 1501; acc:   0.00; ppl:  18.75; 3203 src tok/s; 4198 tgt tok/s;   5700 s elapsed
Epoch  7,  1200/ 1501; acc:   0.00; ppl:  20.17; 3172 src tok/s; 4005 tgt tok/s;   5730 s elapsed
Epoch  7,  1250/ 1501; acc:   0.00; ppl:  19.25; 3037 src tok/s; 3981 tgt tok/s;   5758 s elapsed
Epoch  7,  1300/ 1501; acc:   0.00; ppl:  19.15; 3251 src tok/s; 4186 tgt tok/s;   5786 s elapsed
Epoch  7,  1350/ 1501; acc:   0.00; ppl:  18.73; 3073 src tok/s; 3976 tgt tok/s;   5813 s elapsed
Epoch  7,  1400/ 1501; acc:   0.00; ppl:  18.26; 3164 src tok/s; 4155 tgt tok/s;   5839 s elapsed
Epoch  7,  1450/ 1501; acc:   0.00; ppl:  19.53; 3321 src tok/s; 4245 tgt tok/s;   5869 s elapsed
Epoch  7,  1500/ 1501; acc:   0.00; ppl:  19.00; 3299 src tok/s; 4236 tgt tok/s;   5897 s elapsed
Train perplexity: 19.1298
Train accuracy: 0
Validation perplexity: 27.2835
Validation accuracy: 0

Epoch  8,    50/ 1501; acc:   0.00; ppl:  17.22; 3266 src tok/s; 4208 tgt tok/s;   5928 s elapsed
Epoch  8,   100/ 1501; acc:   0.00; ppl:  17.82; 3314 src tok/s; 4180 tgt tok/s;   5957 s elapsed
Epoch  8,   150/ 1501; acc:   0.00; ppl:  16.04; 3140 src tok/s; 4085 tgt tok/s;   5985 s elapsed
Epoch  8,   200/ 1501; acc:   0.00; ppl:  16.27; 3163 src tok/s; 4165 tgt tok/s;   6011 s elapsed
Epoch  8,   250/ 1501; acc:   0.00; ppl:  16.96; 3148 src tok/s; 4065 tgt tok/s;   6041 s elapsed
Epoch  8,   300/ 1501; acc:   0.00; ppl:  16.26; 3095 src tok/s; 3997 tgt tok/s;   6068 s elapsed
Epoch  8,   350/ 1501; acc:   0.00; ppl:  17.46; 3263 src tok/s; 4180 tgt tok/s;   6096 s elapsed
Epoch  8,   400/ 1501; acc:   0.00; ppl:  17.79; 3258 src tok/s; 4220 tgt tok/s;   6125 s elapsed
Epoch  8,   450/ 1501; acc:   0.00; ppl:  18.03; 3200 src tok/s; 4088 tgt tok/s;   6155 s elapsed
Epoch  8,   500/ 1501; acc:   0.00; ppl:  17.20; 3276 src tok/s; 4235 tgt tok/s;   6184 s elapsed
Epoch  8,   550/ 1501; acc:   0.00; ppl:  16.01; 3155 src tok/s; 4163 tgt tok/s;   6211 s elapsed
Epoch  8,   600/ 1501; acc:   0.00; ppl:  16.92; 3275 src tok/s; 4242 tgt tok/s;   6237 s elapsed
Epoch  8,   650/ 1501; acc:   0.00; ppl:  17.60; 3176 src tok/s; 4075 tgt tok/s;   6267 s elapsed
Epoch  8,   700/ 1501; acc:   0.00; ppl:  17.91; 3247 src tok/s; 4177 tgt tok/s;   6297 s elapsed
Epoch  8,   750/ 1501; acc:   0.00; ppl:  16.85; 3192 src tok/s; 4161 tgt tok/s;   6323 s elapsed
Epoch  8,   800/ 1501; acc:   0.00; ppl:  16.78; 3175 src tok/s; 4124 tgt tok/s;   6349 s elapsed
Epoch  8,   850/ 1501; acc:   0.00; ppl:  17.40; 3218 src tok/s; 4207 tgt tok/s;   6377 s elapsed
Epoch  8,   900/ 1501; acc:   0.00; ppl:  17.14; 3305 src tok/s; 4265 tgt tok/s;   6405 s elapsed
Epoch  8,   950/ 1501; acc:   0.00; ppl:  16.49; 3074 src tok/s; 4033 tgt tok/s;   6432 s elapsed
Epoch  8,  1000/ 1501; acc:   0.00; ppl:  16.94; 3109 src tok/s; 4058 tgt tok/s;   6459 s elapsed
Epoch  8,  1050/ 1501; acc:   0.00; ppl:  19.33; 3346 src tok/s; 4252 tgt tok/s;   6489 s elapsed
Epoch  8,  1100/ 1501; acc:   0.00; ppl:  17.36; 3057 src tok/s; 3971 tgt tok/s;   6519 s elapsed
Epoch  8,  1150/ 1501; acc:   0.00; ppl:  16.69; 3179 src tok/s; 4156 tgt tok/s;   6549 s elapsed
Epoch  8,  1200/ 1501; acc:   0.00; ppl:  16.35; 2960 src tok/s; 3866 tgt tok/s;   6576 s elapsed
Epoch  8,  1250/ 1501; acc:   0.00; ppl:  18.18; 3195 src tok/s; 4083 tgt tok/s;   6606 s elapsed
Epoch  8,  1300/ 1501; acc:   0.00; ppl:  16.78; 2985 src tok/s; 3866 tgt tok/s;   6635 s elapsed
Epoch  8,  1350/ 1501; acc:   0.00; ppl:  16.85; 3159 src tok/s; 4109 tgt tok/s;   6663 s elapsed
Epoch  8,  1400/ 1501; acc:   0.00; ppl:  16.87; 2974 src tok/s; 3860 tgt tok/s;   6691 s elapsed
Epoch  8,  1450/ 1501; acc:   0.00; ppl:  17.06; 3236 src tok/s; 4208 tgt tok/s;   6718 s elapsed
Epoch  8,  1500/ 1501; acc:   0.00; ppl:  16.79; 3004 src tok/s; 3916 tgt tok/s;   6748 s elapsed
Train perplexity: 17.1219
Train accuracy: 0
Validation perplexity: 24.2612
Validation accuracy: 0
Decaying learning rate to 0.5

Epoch  9,    50/ 1501; acc:   0.00; ppl:  14.02; 3113 src tok/s; 4050 tgt tok/s;   6778 s elapsed
Epoch  9,   100/ 1501; acc:   0.00; ppl:  14.28; 3183 src tok/s; 4101 tgt tok/s;   6807 s elapsed
Epoch  9,   150/ 1501; acc:   0.00; ppl:  13.56; 3043 src tok/s; 3987 tgt tok/s;   6836 s elapsed
Epoch  9,   200/ 1501; acc:   0.00; ppl:  14.58; 3303 src tok/s; 4238 tgt tok/s;   6866 s elapsed
Epoch  9,   250/ 1501; acc:   0.00; ppl:  14.09; 3362 src tok/s; 4348 tgt tok/s;   6895 s elapsed
Epoch  9,   300/ 1501; acc:   0.00; ppl:  13.52; 3273 src tok/s; 4233 tgt tok/s;   6922 s elapsed
Epoch  9,   350/ 1501; acc:   0.00; ppl:  14.49; 3209 src tok/s; 4113 tgt tok/s;   6951 s elapsed
Epoch  9,   400/ 1501; acc:   0.00; ppl:  13.73; 3111 src tok/s; 4020 tgt tok/s;   6979 s elapsed
Epoch  9,   450/ 1501; acc:   0.00; ppl:  14.04; 3225 src tok/s; 4154 tgt tok/s;   7009 s elapsed
Epoch  9,   500/ 1501; acc:   0.00; ppl:  13.00; 3074 src tok/s; 4015 tgt tok/s;   7035 s elapsed
Epoch  9,   550/ 1501; acc:   0.00; ppl:  13.61; 3231 src tok/s; 4154 tgt tok/s;   7063 s elapsed
Epoch  9,   600/ 1501; acc:   0.00; ppl:  13.97; 3073 src tok/s; 3931 tgt tok/s;   7093 s elapsed
Epoch  9,   650/ 1501; acc:   0.00; ppl:  14.25; 3144 src tok/s; 4051 tgt tok/s;   7123 s elapsed
Epoch  9,   700/ 1501; acc:   0.00; ppl:  13.53; 3047 src tok/s; 3964 tgt tok/s;   7154 s elapsed
Epoch  9,   750/ 1501; acc:   0.00; ppl:  13.22; 2976 src tok/s; 3906 tgt tok/s;   7182 s elapsed
Epoch  9,   800/ 1501; acc:   0.00; ppl:  13.86; 3092 src tok/s; 4042 tgt tok/s;   7213 s elapsed
Epoch  9,   850/ 1501; acc:   0.00; ppl:  13.64; 3201 src tok/s; 4144 tgt tok/s;   7241 s elapsed
Epoch  9,   900/ 1501; acc:   0.00; ppl:  14.39; 3299 src tok/s; 4199 tgt tok/s;   7268 s elapsed
Epoch  9,   950/ 1501; acc:   0.00; ppl:  13.35; 3168 src tok/s; 4083 tgt tok/s;   7296 s elapsed
Epoch  9,  1000/ 1501; acc:   0.00; ppl:  12.86; 3163 src tok/s; 4121 tgt tok/s;   7322 s elapsed
Epoch  9,  1050/ 1501; acc:   0.00; ppl:  13.34; 3218 src tok/s; 4188 tgt tok/s;   7348 s elapsed
Epoch  9,  1100/ 1501; acc:   0.00; ppl:  12.82; 3190 src tok/s; 4127 tgt tok/s;   7375 s elapsed
Epoch  9,  1150/ 1501; acc:   0.00; ppl:  12.90; 3048 src tok/s; 4000 tgt tok/s;   7402 s elapsed
Epoch  9,  1200/ 1501; acc:   0.00; ppl:  13.11; 2862 src tok/s; 3765 tgt tok/s;   7430 s elapsed
Epoch  9,  1250/ 1501; acc:   0.00; ppl:  12.98; 2901 src tok/s; 3805 tgt tok/s;   7461 s elapsed
Epoch  9,  1300/ 1501; acc:   0.00; ppl:  13.70; 3066 src tok/s; 3935 tgt tok/s;   7489 s elapsed
Epoch  9,  1350/ 1501; acc:   0.00; ppl:  13.57; 3138 src tok/s; 4062 tgt tok/s;   7518 s elapsed
Epoch  9,  1400/ 1501; acc:   0.00; ppl:  13.07; 3181 src tok/s; 4106 tgt tok/s;   7545 s elapsed
Epoch  9,  1450/ 1501; acc:   0.00; ppl:  13.54; 3255 src tok/s; 4222 tgt tok/s;   7572 s elapsed
Epoch  9,  1500/ 1501; acc:   0.00; ppl:  14.64; 3383 src tok/s; 4331 tgt tok/s;   7604 s elapsed
Train perplexity: 13.6665
Train accuracy: 0
Validation perplexity: 21.0879
Validation accuracy: 0
Decaying learning rate to 0.25

Epoch 10,    50/ 1501; acc:   0.00; ppl:  11.31; 3066 src tok/s; 4029 tgt tok/s;   7634 s elapsed
Epoch 10,   100/ 1501; acc:   0.00; ppl:  12.73; 3198 src tok/s; 4050 tgt tok/s;   7664 s elapsed
Epoch 10,   150/ 1501; acc:   0.00; ppl:  11.55; 3112 src tok/s; 4043 tgt tok/s;   7691 s elapsed
Epoch 10,   200/ 1501; acc:   0.00; ppl:  12.06; 3031 src tok/s; 3929 tgt tok/s;   7722 s elapsed
Epoch 10,   250/ 1501; acc:   0.00; ppl:  12.21; 3047 src tok/s; 3934 tgt tok/s;   7754 s elapsed
Epoch 10,   300/ 1501; acc:   0.00; ppl:  12.46; 2961 src tok/s; 3794 tgt tok/s;   7785 s elapsed
Epoch 10,   350/ 1501; acc:   0.00; ppl:  11.69; 2979 src tok/s; 3865 tgt tok/s;   7814 s elapsed
Epoch 10,   400/ 1501; acc:   0.00; ppl:  11.72; 2953 src tok/s; 3836 tgt tok/s;   7845 s elapsed
Epoch 10,   450/ 1501; acc:   0.00; ppl:  11.25; 2819 src tok/s; 3702 tgt tok/s;   7873 s elapsed
Epoch 10,   500/ 1501; acc:   0.00; ppl:  11.69; 3028 src tok/s; 3908 tgt tok/s;   7904 s elapsed
Epoch 10,   550/ 1501; acc:   0.00; ppl:  12.42; 3095 src tok/s; 3957 tgt tok/s;   7933 s elapsed
Epoch 10,   600/ 1501; acc:   0.00; ppl:  11.63; 3021 src tok/s; 3918 tgt tok/s;   7962 s elapsed
Epoch 10,   650/ 1501; acc:   0.00; ppl:  11.53; 3012 src tok/s; 3883 tgt tok/s;   7990 s elapsed
Epoch 10,   700/ 1501; acc:   0.00; ppl:  11.98; 3045 src tok/s; 3893 tgt tok/s;   8020 s elapsed
Epoch 10,   750/ 1501; acc:   0.00; ppl:  12.26; 3000 src tok/s; 3886 tgt tok/s;   8050 s elapsed
Epoch 10,   800/ 1501; acc:   0.00; ppl:  11.42; 3098 src tok/s; 4065 tgt tok/s;   8078 s elapsed
Epoch 10,   850/ 1501; acc:   0.00; ppl:  11.35; 3146 src tok/s; 4137 tgt tok/s;   8103 s elapsed
Epoch 10,   900/ 1501; acc:   0.00; ppl:  12.67; 3254 src tok/s; 4181 tgt tok/s;   8133 s elapsed
Epoch 10,   950/ 1501; acc:   0.00; ppl:  11.30; 3087 src tok/s; 4022 tgt tok/s;   8160 s elapsed
Epoch 10,  1000/ 1501; acc:   0.00; ppl:  12.10; 3217 src tok/s; 4171 tgt tok/s;   8188 s elapsed
Epoch 10,  1050/ 1501; acc:   0.00; ppl:  12.08; 3194 src tok/s; 4164 tgt tok/s;   8218 s elapsed
Epoch 10,  1100/ 1501; acc:   0.00; ppl:  11.77; 3240 src tok/s; 4246 tgt tok/s;   8246 s elapsed
Epoch 10,  1150/ 1501; acc:   0.00; ppl:  12.40; 3257 src tok/s; 4169 tgt tok/s;   8276 s elapsed
Epoch 10,  1200/ 1501; acc:   0.00; ppl:  12.04; 3228 src tok/s; 4186 tgt tok/s;   8305 s elapsed
Epoch 10,  1250/ 1501; acc:   0.00; ppl:  12.21; 3310 src tok/s; 4259 tgt tok/s;   8332 s elapsed
Epoch 10,  1300/ 1501; acc:   0.00; ppl:  11.54; 3188 src tok/s; 4116 tgt tok/s;   8360 s elapsed
Epoch 10,  1350/ 1501; acc:   0.00; ppl:  11.82; 3197 src tok/s; 4125 tgt tok/s;   8389 s elapsed
Epoch 10,  1400/ 1501; acc:   0.00; ppl:  11.90; 3260 src tok/s; 4245 tgt tok/s;   8416 s elapsed
Epoch 10,  1450/ 1501; acc:   0.00; ppl:  12.12; 3184 src tok/s; 4115 tgt tok/s;   8444 s elapsed
Epoch 10,  1500/ 1501; acc:   0.00; ppl:  11.34; 3078 src tok/s; 4018 tgt tok/s;   8471 s elapsed
Train perplexity: 11.8957
Train accuracy: 0
Validation perplexity: 19.9773
Validation accuracy: 0
Decaying learning rate to 0.125

Epoch 11,    50/ 1501; acc:   0.00; ppl:  11.18; 3242 src tok/s; 4190 tgt tok/s;   8501 s elapsed
Epoch 11,   100/ 1501; acc:   0.00; ppl:  11.11; 3270 src tok/s; 4236 tgt tok/s;   8529 s elapsed
Epoch 11,   150/ 1501; acc:   0.00; ppl:  11.78; 3097 src tok/s; 3987 tgt tok/s;   8560 s elapsed
Epoch 11,   200/ 1501; acc:   0.00; ppl:  10.93; 3090 src tok/s; 4014 tgt tok/s;   8589 s elapsed
Epoch 11,   250/ 1501; acc:   0.00; ppl:  11.83; 3393 src tok/s; 4300 tgt tok/s;   8621 s elapsed
Epoch 11,   300/ 1501; acc:   0.00; ppl:  10.26; 3281 src tok/s; 4264 tgt tok/s;   8647 s elapsed
Epoch 11,   350/ 1501; acc:   0.00; ppl:  11.28; 3274 src tok/s; 4198 tgt tok/s;   8677 s elapsed
Epoch 11,   400/ 1501; acc:   0.00; ppl:  10.93; 3109 src tok/s; 4061 tgt tok/s;   8704 s elapsed
Epoch 11,   450/ 1501; acc:   0.00; ppl:  10.89; 3069 src tok/s; 3988 tgt tok/s;   8732 s elapsed
Epoch 11,   500/ 1501; acc:   0.00; ppl:  10.57; 3143 src tok/s; 4089 tgt tok/s;   8760 s elapsed
Epoch 11,   550/ 1501; acc:   0.00; ppl:  10.39; 2982 src tok/s; 3906 tgt tok/s;   8788 s elapsed
Epoch 11,   600/ 1501; acc:   0.00; ppl:  10.88; 3121 src tok/s; 4028 tgt tok/s;   8815 s elapsed
Epoch 11,   650/ 1501; acc:   0.00; ppl:  11.35; 3229 src tok/s; 4185 tgt tok/s;   8843 s elapsed
Epoch 11,   700/ 1501; acc:   0.00; ppl:  11.31; 3223 src tok/s; 4141 tgt tok/s;   8872 s elapsed
Epoch 11,   750/ 1501; acc:   0.00; ppl:  11.40; 3222 src tok/s; 4180 tgt tok/s;   8901 s elapsed
Epoch 11,   800/ 1501; acc:   0.00; ppl:  10.82; 3200 src tok/s; 4132 tgt tok/s;   8929 s elapsed
Epoch 11,   850/ 1501; acc:   0.00; ppl:  11.14; 3225 src tok/s; 4187 tgt tok/s;   8956 s elapsed
Epoch 11,   900/ 1501; acc:   0.00; ppl:  10.72; 3082 src tok/s; 4011 tgt tok/s;   8983 s elapsed
Epoch 11,   950/ 1501; acc:   0.00; ppl:  11.09; 3213 src tok/s; 4116 tgt tok/s;   9012 s elapsed
Epoch 11,  1000/ 1501; acc:   0.00; ppl:  11.10; 3278 src tok/s; 4238 tgt tok/s;   9041 s elapsed
Epoch 11,  1050/ 1501; acc:   0.00; ppl:  10.99; 3231 src tok/s; 4202 tgt tok/s;   9068 s elapsed
Epoch 11,  1100/ 1501; acc:   0.00; ppl:  11.51; 3252 src tok/s; 4203 tgt tok/s;   9097 s elapsed
Epoch 11,  1150/ 1501; acc:   0.00; ppl:  11.44; 3262 src tok/s; 4172 tgt tok/s;   9127 s elapsed
Epoch 11,  1200/ 1501; acc:   0.00; ppl:  10.96; 3221 src tok/s; 4164 tgt tok/s;   9153 s elapsed
Epoch 11,  1250/ 1501; acc:   0.00; ppl:  10.55; 3112 src tok/s; 4077 tgt tok/s;   9180 s elapsed
Epoch 11,  1300/ 1501; acc:   0.00; ppl:  10.21; 3037 src tok/s; 4024 tgt tok/s;   9205 s elapsed
Epoch 11,  1350/ 1501; acc:   0.00; ppl:  11.46; 3257 src tok/s; 4153 tgt tok/s;   9234 s elapsed
Epoch 11,  1400/ 1501; acc:   0.00; ppl:  10.45; 3055 src tok/s; 4028 tgt tok/s;   9262 s elapsed
Epoch 11,  1450/ 1501; acc:   0.00; ppl:  11.18; 3310 src tok/s; 4280 tgt tok/s;   9289 s elapsed
Epoch 11,  1500/ 1501; acc:   0.00; ppl:  10.97; 3146 src tok/s; 4091 tgt tok/s;   9317 s elapsed
Train perplexity: 11.0333
Train accuracy: 0
Validation perplexity: 19.3675
Validation accuracy: 0
Decaying learning rate to 0.0625

Epoch 12,    50/ 1501; acc:   0.00; ppl:  10.33; 2992 src tok/s; 3949 tgt tok/s;   9347 s elapsed
Epoch 12,   100/ 1501; acc:   0.00; ppl:   9.93; 2917 src tok/s; 3804 tgt tok/s;   9375 s elapsed
Epoch 12,   150/ 1501; acc:   0.00; ppl:   9.40; 3128 src tok/s; 4152 tgt tok/s;   9398 s elapsed
Epoch 12,   200/ 1501; acc:   0.00; ppl:  10.66; 3321 src tok/s; 4255 tgt tok/s;   9427 s elapsed
Epoch 12,   250/ 1501; acc:   0.00; ppl:  10.63; 3231 src tok/s; 4187 tgt tok/s;   9455 s elapsed
Epoch 12,   300/ 1501; acc:   0.00; ppl:  10.88; 3286 src tok/s; 4246 tgt tok/s;   9484 s elapsed
Epoch 12,   350/ 1501; acc:   0.00; ppl:  10.50; 3333 src tok/s; 4300 tgt tok/s;   9511 s elapsed
Epoch 12,   400/ 1501; acc:   0.00; ppl:  10.17; 3260 src tok/s; 4266 tgt tok/s;   9537 s elapsed
Epoch 12,   450/ 1501; acc:   0.00; ppl:  11.13; 3452 src tok/s; 4380 tgt tok/s;   9565 s elapsed
Epoch 12,   500/ 1501; acc:   0.00; ppl:  10.44; 3202 src tok/s; 4167 tgt tok/s;   9592 s elapsed
Epoch 12,   550/ 1501; acc:   0.00; ppl:  10.72; 3179 src tok/s; 4143 tgt tok/s;   9620 s elapsed
Epoch 12,   600/ 1501; acc:   0.00; ppl:  10.44; 3198 src tok/s; 4169 tgt tok/s;   9648 s elapsed
Epoch 12,   650/ 1501; acc:   0.00; ppl:  11.07; 3264 src tok/s; 4216 tgt tok/s;   9676 s elapsed
Epoch 12,   700/ 1501; acc:   0.00; ppl:  11.10; 3301 src tok/s; 4233 tgt tok/s;   9706 s elapsed
Epoch 12,   750/ 1501; acc:   0.00; ppl:  10.16; 3104 src tok/s; 4049 tgt tok/s;   9732 s elapsed
Epoch 12,   800/ 1501; acc:   0.00; ppl:  10.29; 3121 src tok/s; 4080 tgt tok/s;   9757 s elapsed
Epoch 12,   850/ 1501; acc:   0.00; ppl:  10.36; 3177 src tok/s; 4146 tgt tok/s;   9785 s elapsed
Epoch 12,   900/ 1501; acc:   0.00; ppl:  11.02; 3298 src tok/s; 4191 tgt tok/s;   9814 s elapsed
Epoch 12,   950/ 1501; acc:   0.00; ppl:  10.91; 3272 src tok/s; 4234 tgt tok/s;   9844 s elapsed
Epoch 12,  1000/ 1501; acc:   0.00; ppl:  10.58; 3186 src tok/s; 4137 tgt tok/s;   9872 s elapsed
Epoch 12,  1050/ 1501; acc:   0.00; ppl:  10.40; 3299 src tok/s; 4260 tgt tok/s;   9900 s elapsed
Epoch 12,  1100/ 1501; acc:   0.00; ppl:  10.26; 3125 src tok/s; 4123 tgt tok/s;   9928 s elapsed
Epoch 12,  1150/ 1501; acc:   0.00; ppl:  10.38; 3200 src tok/s; 4141 tgt tok/s;   9955 s elapsed
Epoch 12,  1200/ 1501; acc:   0.00; ppl:  10.78; 3342 src tok/s; 4303 tgt tok/s;   9981 s elapsed
Epoch 12,  1250/ 1501; acc:   0.00; ppl:  11.07; 3355 src tok/s; 4276 tgt tok/s;  10010 s elapsed
Epoch 12,  1300/ 1501; acc:   0.00; ppl:  10.71; 3237 src tok/s; 4189 tgt tok/s;  10039 s elapsed
Epoch 12,  1350/ 1501; acc:   0.00; ppl:  10.65; 3220 src tok/s; 4193 tgt tok/s;  10067 s elapsed
Epoch 12,  1400/ 1501; acc:   0.00; ppl:  11.24; 3303 src tok/s; 4234 tgt tok/s;  10097 s elapsed
Epoch 12,  1450/ 1501; acc:   0.00; ppl:  10.72; 3310 src tok/s; 4265 tgt tok/s;  10123 s elapsed
Epoch 12,  1500/ 1501; acc:   0.00; ppl:  10.86; 3351 src tok/s; 4285 tgt tok/s;  10151 s elapsed
Train perplexity: 10.6041
Train accuracy: 0
Validation perplexity: 19.2151
Validation accuracy: 0
Decaying learning rate to 0.03125

Epoch 13,    50/ 1501; acc:   0.00; ppl:  10.13; 3227 src tok/s; 4208 tgt tok/s;  10182 s elapsed
Epoch 13,   100/ 1501; acc:   0.00; ppl:  10.84; 3400 src tok/s; 4349 tgt tok/s;  10211 s elapsed
Epoch 13,   150/ 1501; acc:   0.00; ppl:  10.39; 3221 src tok/s; 4184 tgt tok/s;  10239 s elapsed
Epoch 13,   200/ 1501; acc:   0.00; ppl:  10.50; 3210 src tok/s; 4125 tgt tok/s;  10267 s elapsed
Epoch 13,   250/ 1501; acc:   0.00; ppl:  10.65; 3224 src tok/s; 4184 tgt tok/s;  10296 s elapsed
Epoch 13,   300/ 1501; acc:   0.00; ppl:  10.41; 3152 src tok/s; 4106 tgt tok/s;  10325 s elapsed
Epoch 13,   350/ 1501; acc:   0.00; ppl:  10.41; 3072 src tok/s; 3972 tgt tok/s;  10357 s elapsed
Epoch 13,   400/ 1501; acc:   0.00; ppl:   9.87; 2991 src tok/s; 3913 tgt tok/s;  10383 s elapsed
Epoch 13,   450/ 1501; acc:   0.00; ppl:  10.26; 3080 src tok/s; 3980 tgt tok/s;  10412 s elapsed
Epoch 13,   500/ 1501; acc:   0.00; ppl:  11.56; 3380 src tok/s; 4240 tgt tok/s;  10444 s elapsed
Epoch 13,   550/ 1501; acc:   0.00; ppl:  10.26; 3209 src tok/s; 4142 tgt tok/s;  10473 s elapsed
Epoch 13,   600/ 1501; acc:   0.00; ppl:  10.43; 3349 src tok/s; 4285 tgt tok/s;  10501 s elapsed
Epoch 13,   650/ 1501; acc:   0.00; ppl:   9.89; 3207 src tok/s; 4207 tgt tok/s;  10527 s elapsed
Epoch 13,   700/ 1501; acc:   0.00; ppl:  10.99; 3358 src tok/s; 4338 tgt tok/s;  10557 s elapsed
Epoch 13,   750/ 1501; acc:   0.00; ppl:  10.00; 3140 src tok/s; 4077 tgt tok/s;  10583 s elapsed
Epoch 13,   800/ 1501; acc:   0.00; ppl:  10.19; 3153 src tok/s; 4167 tgt tok/s;  10610 s elapsed
Epoch 13,   850/ 1501; acc:   0.00; ppl:  10.60; 3166 src tok/s; 4111 tgt tok/s;  10639 s elapsed
Epoch 13,   900/ 1501; acc:   0.00; ppl:  10.29; 3115 src tok/s; 4025 tgt tok/s;  10666 s elapsed
Epoch 13,   950/ 1501; acc:   0.00; ppl:  10.27; 3213 src tok/s; 4145 tgt tok/s;  10693 s elapsed
Epoch 13,  1000/ 1501; acc:   0.00; ppl:   9.51; 3008 src tok/s; 3989 tgt tok/s;  10719 s elapsed
Epoch 13,  1050/ 1501; acc:   0.00; ppl:  10.50; 3262 src tok/s; 4236 tgt tok/s;  10747 s elapsed
Epoch 13,  1100/ 1501; acc:   0.00; ppl:  10.07; 3175 src tok/s; 4156 tgt tok/s;  10774 s elapsed
Epoch 13,  1150/ 1501; acc:   0.00; ppl:  10.35; 3025 src tok/s; 3996 tgt tok/s;  10801 s elapsed
Epoch 13,  1200/ 1501; acc:   0.00; ppl:  10.51; 3271 src tok/s; 4181 tgt tok/s;  10828 s elapsed
Epoch 13,  1250/ 1501; acc:   0.00; ppl:  10.47; 3162 src tok/s; 4072 tgt tok/s;  10857 s elapsed
Epoch 13,  1300/ 1501; acc:   0.00; ppl:  11.27; 3308 src tok/s; 4175 tgt tok/s;  10888 s elapsed
Epoch 13,  1350/ 1501; acc:   0.00; ppl:   9.65; 3008 src tok/s; 3921 tgt tok/s;  10914 s elapsed
Epoch 13,  1400/ 1501; acc:   0.00; ppl:  10.61; 3296 src tok/s; 4215 tgt tok/s;  10942 s elapsed
Epoch 13,  1450/ 1501; acc:   0.00; ppl:   9.99; 2937 src tok/s; 3849 tgt tok/s;  10971 s elapsed
Epoch 13,  1500/ 1501; acc:   0.00; ppl:  10.22; 2980 src tok/s; 3886 tgt tok/s;  11000 s elapsed
Train perplexity: 10.3867
Train accuracy: 0
Validation perplexity: 19.0987
Validation accuracy: 0
Decaying learning rate to 0.015625

Epoch 14,    50/ 1501; acc:   0.00; ppl:   9.96; 3006 src tok/s; 3944 tgt tok/s;  11032 s elapsed
Epoch 14,   100/ 1501; acc:   0.00; ppl:   9.99; 3162 src tok/s; 4129 tgt tok/s;  11061 s elapsed
Epoch 14,   150/ 1501; acc:   0.00; ppl:  10.88; 3288 src tok/s; 4210 tgt tok/s;  11091 s elapsed
Epoch 14,   200/ 1501; acc:   0.00; ppl:  10.46; 3136 src tok/s; 4035 tgt tok/s;  11121 s elapsed
Epoch 14,   250/ 1501; acc:   0.00; ppl:  10.20; 3072 src tok/s; 3949 tgt tok/s;  11151 s elapsed
Epoch 14,   300/ 1501; acc:   0.00; ppl:  10.23; 3035 src tok/s; 3957 tgt tok/s;  11181 s elapsed
Epoch 14,   350/ 1501; acc:   0.00; ppl:  10.29; 3121 src tok/s; 3999 tgt tok/s;  11210 s elapsed
Epoch 14,   400/ 1501; acc:   0.00; ppl:  11.02; 3129 src tok/s; 3993 tgt tok/s;  11241 s elapsed
Epoch 14,   450/ 1501; acc:   0.00; ppl:  10.91; 3073 src tok/s; 3890 tgt tok/s;  11271 s elapsed
Epoch 14,   500/ 1501; acc:   0.00; ppl:   9.79; 2884 src tok/s; 3787 tgt tok/s;  11300 s elapsed
Epoch 14,   550/ 1501; acc:   0.00; ppl:   9.68; 2994 src tok/s; 3934 tgt tok/s;  11328 s elapsed
Epoch 14,   600/ 1501; acc:   0.00; ppl:  10.36; 3038 src tok/s; 3944 tgt tok/s;  11357 s elapsed
Epoch 14,   650/ 1501; acc:   0.00; ppl:   9.93; 3008 src tok/s; 3951 tgt tok/s;  11384 s elapsed
Epoch 14,   700/ 1501; acc:   0.00; ppl:  10.58; 3240 src tok/s; 4165 tgt tok/s;  11413 s elapsed
Epoch 14,   750/ 1501; acc:   0.00; ppl:  10.76; 3194 src tok/s; 4077 tgt tok/s;  11445 s elapsed
Epoch 14,   800/ 1501; acc:   0.00; ppl:  10.52; 3132 src tok/s; 4054 tgt tok/s;  11476 s elapsed
Epoch 14,   850/ 1501; acc:   0.00; ppl:  10.95; 3183 src tok/s; 4067 tgt tok/s;  11507 s elapsed
Epoch 14,   900/ 1501; acc:   0.00; ppl:   9.77; 3021 src tok/s; 4003 tgt tok/s;  11533 s elapsed
Epoch 14,   950/ 1501; acc:   0.00; ppl:  10.61; 3189 src tok/s; 4118 tgt tok/s;  11563 s elapsed
Epoch 14,  1000/ 1501; acc:   0.00; ppl:  10.78; 3343 src tok/s; 4262 tgt tok/s;  11592 s elapsed
Epoch 14,  1050/ 1501; acc:   0.00; ppl:   9.38; 3029 src tok/s; 4001 tgt tok/s;  11617 s elapsed
Epoch 14,  1100/ 1501; acc:   0.00; ppl:   9.92; 3112 src tok/s; 4049 tgt tok/s;  11644 s elapsed
Epoch 14,  1150/ 1501; acc:   0.00; ppl:   9.70; 3119 src tok/s; 4092 tgt tok/s;  11671 s elapsed
Epoch 14,  1200/ 1501; acc:   0.00; ppl:  10.50; 3180 src tok/s; 4079 tgt tok/s;  11699 s elapsed
Epoch 14,  1250/ 1501; acc:   0.00; ppl:  10.62; 3241 src tok/s; 4190 tgt tok/s;  11728 s elapsed
Epoch 14,  1300/ 1501; acc:   0.00; ppl:  10.10; 3228 src tok/s; 4227 tgt tok/s;  11754 s elapsed
Epoch 14,  1350/ 1501; acc:   0.00; ppl:  10.18; 3189 src tok/s; 4098 tgt tok/s;  11781 s elapsed
Epoch 14,  1400/ 1501; acc:   0.00; ppl:   9.49; 3068 src tok/s; 4053 tgt tok/s;  11807 s elapsed
Epoch 14,  1450/ 1501; acc:   0.00; ppl:  10.72; 3275 src tok/s; 4190 tgt tok/s;  11835 s elapsed
Epoch 14,  1500/ 1501; acc:   0.00; ppl:   9.90; 3060 src tok/s; 3998 tgt tok/s;  11863 s elapsed
Train perplexity: 10.2881
Train accuracy: 0
Validation perplexity: 19.0485
Validation accuracy: 0
Decaying learning rate to 0.0078125

Epoch 15,    50/ 1501; acc:   0.00; ppl:  10.38; 3199 src tok/s; 4115 tgt tok/s;  11894 s elapsed
Epoch 15,   100/ 1501; acc:   0.00; ppl:  10.78; 3244 src tok/s; 4156 tgt tok/s;  11924 s elapsed
Epoch 15,   150/ 1501; acc:   0.00; ppl:  10.58; 3181 src tok/s; 4126 tgt tok/s;  11954 s elapsed
Epoch 15,   200/ 1501; acc:   0.00; ppl:  10.50; 3155 src tok/s; 4056 tgt tok/s;  11984 s elapsed
Epoch 15,   250/ 1501; acc:   0.00; ppl:  10.16; 3154 src tok/s; 4086 tgt tok/s;  12011 s elapsed
Epoch 15,   300/ 1501; acc:   0.00; ppl:   9.79; 2826 src tok/s; 3696 tgt tok/s;  12041 s elapsed
Epoch 15,   350/ 1501; acc:   0.00; ppl:  10.45; 3087 src tok/s; 3938 tgt tok/s;  12070 s elapsed
Epoch 15,   400/ 1501; acc:   0.00; ppl:  10.64; 3034 src tok/s; 3888 tgt tok/s;  12101 s elapsed
Epoch 15,   450/ 1501; acc:   0.00; ppl:  10.22; 3223 src tok/s; 4149 tgt tok/s;  12128 s elapsed
Epoch 15,   500/ 1501; acc:   0.00; ppl:   9.86; 3254 src tok/s; 4204 tgt tok/s;  12154 s elapsed
Epoch 15,   550/ 1501; acc:   0.00; ppl:  10.33; 3133 src tok/s; 4038 tgt tok/s;  12182 s elapsed
Epoch 15,   600/ 1501; acc:   0.00; ppl:   9.69; 3136 src tok/s; 4129 tgt tok/s;  12208 s elapsed
Epoch 15,   650/ 1501; acc:   0.00; ppl:  10.32; 3024 src tok/s; 3903 tgt tok/s;  12239 s elapsed
Epoch 15,   700/ 1501; acc:   0.00; ppl:  10.00; 3047 src tok/s; 3938 tgt tok/s;  12268 s elapsed
Epoch 15,   750/ 1501; acc:   0.00; ppl:  10.20; 3196 src tok/s; 4180 tgt tok/s;  12296 s elapsed
Epoch 15,   800/ 1501; acc:   0.00; ppl:  10.13; 3220 src tok/s; 4189 tgt tok/s;  12324 s elapsed
Epoch 15,   850/ 1501; acc:   0.00; ppl:   9.61; 2961 src tok/s; 3901 tgt tok/s;  12353 s elapsed
Epoch 15,   900/ 1501; acc:   0.00; ppl:   9.71; 2993 src tok/s; 3932 tgt tok/s;  12380 s elapsed
Epoch 15,   950/ 1501; acc:   0.00; ppl:  10.80; 3344 src tok/s; 4271 tgt tok/s;  12409 s elapsed
Epoch 15,  1000/ 1501; acc:   0.00; ppl:  10.28; 3292 src tok/s; 4235 tgt tok/s;  12437 s elapsed
Epoch 15,  1050/ 1501; acc:   0.00; ppl:   9.76; 3226 src tok/s; 4214 tgt tok/s;  12464 s elapsed
Epoch 15,  1100/ 1501; acc:   0.00; ppl:   9.95; 3207 src tok/s; 4166 tgt tok/s;  12491 s elapsed
Epoch 15,  1150/ 1501; acc:   0.00; ppl:  10.24; 3190 src tok/s; 4157 tgt tok/s;  12519 s elapsed
Epoch 15,  1200/ 1501; acc:   0.00; ppl:   9.92; 3292 src tok/s; 4272 tgt tok/s;  12546 s elapsed
Epoch 15,  1250/ 1501; acc:   0.00; ppl:  10.39; 3169 src tok/s; 4102 tgt tok/s;  12574 s elapsed
Epoch 15,  1300/ 1501; acc:   0.00; ppl:  10.93; 3451 src tok/s; 4380 tgt tok/s;  12605 s elapsed
Epoch 15,  1350/ 1501; acc:   0.00; ppl:   9.58; 3076 src tok/s; 4083 tgt tok/s;  12630 s elapsed
Epoch 15,  1400/ 1501; acc:   0.00; ppl:   9.29; 3162 src tok/s; 4201 tgt tok/s;  12655 s elapsed
Epoch 15,  1450/ 1501; acc:   0.00; ppl:  11.20; 3307 src tok/s; 4195 tgt tok/s;  12686 s elapsed
Epoch 15,  1500/ 1501; acc:   0.00; ppl:  10.64; 3084 src tok/s; 3984 tgt tok/s;  12716 s elapsed
Train perplexity: 10.224
Train accuracy: 0
Validation perplexity: 19.047
Validation accuracy: 0
Decaying learning rate to 0.00390625

Epoch 16,    50/ 1501; acc:   0.00; ppl:  10.42; 3270 src tok/s; 4200 tgt tok/s;  12747 s elapsed
Epoch 16,   100/ 1501; acc:   0.00; ppl:  10.21; 3145 src tok/s; 4094 tgt tok/s;  12777 s elapsed
Epoch 16,   150/ 1501; acc:   0.00; ppl:  10.99; 3477 src tok/s; 4421 tgt tok/s;  12806 s elapsed
Epoch 16,   200/ 1501; acc:   0.00; ppl:   9.90; 3174 src tok/s; 4129 tgt tok/s;  12833 s elapsed
Epoch 16,   250/ 1501; acc:   0.00; ppl:  10.41; 3180 src tok/s; 4088 tgt tok/s;  12862 s elapsed
Epoch 16,   300/ 1501; acc:   0.00; ppl:   9.89; 3221 src tok/s; 4182 tgt tok/s;  12889 s elapsed
Epoch 16,   350/ 1501; acc:   0.00; ppl:  10.84; 3382 src tok/s; 4293 tgt tok/s;  12918 s elapsed
Epoch 16,   400/ 1501; acc:   0.00; ppl:   9.98; 3297 src tok/s; 4257 tgt tok/s;  12943 s elapsed
Epoch 16,   450/ 1501; acc:   0.00; ppl:  10.54; 3237 src tok/s; 4153 tgt tok/s;  12972 s elapsed
Epoch 16,   500/ 1501; acc:   0.00; ppl:  10.72; 3294 src tok/s; 4209 tgt tok/s;  13003 s elapsed
Epoch 16,   550/ 1501; acc:   0.00; ppl:   9.62; 3102 src tok/s; 4066 tgt tok/s;  13029 s elapsed
Epoch 16,   600/ 1501; acc:   0.00; ppl:  10.80; 3444 src tok/s; 4399 tgt tok/s;  13056 s elapsed
Epoch 16,   650/ 1501; acc:   0.00; ppl:  10.58; 3365 src tok/s; 4329 tgt tok/s;  13086 s elapsed
Epoch 16,   700/ 1501; acc:   0.00; ppl:   9.88; 3120 src tok/s; 4044 tgt tok/s;  13113 s elapsed
Epoch 16,   750/ 1501; acc:   0.00; ppl:  10.38; 3138 src tok/s; 4085 tgt tok/s;  13143 s elapsed
Epoch 16,   800/ 1501; acc:   0.00; ppl:   9.82; 2900 src tok/s; 3793 tgt tok/s;  13172 s elapsed
Epoch 16,   850/ 1501; acc:   0.00; ppl:  10.42; 3216 src tok/s; 4158 tgt tok/s;  13200 s elapsed
Epoch 16,   900/ 1501; acc:   0.00; ppl:   9.82; 3151 src tok/s; 4103 tgt tok/s;  13228 s elapsed
Epoch 16,   950/ 1501; acc:   0.00; ppl:  10.21; 3184 src tok/s; 4128 tgt tok/s;  13257 s elapsed
Epoch 16,  1000/ 1501; acc:   0.00; ppl:   9.41; 3008 src tok/s; 3986 tgt tok/s;  13284 s elapsed
Epoch 16,  1050/ 1501; acc:   0.00; ppl:  10.33; 3102 src tok/s; 4021 tgt tok/s;  13312 s elapsed
Epoch 16,  1100/ 1501; acc:   0.00; ppl:   9.95; 3040 src tok/s; 3960 tgt tok/s;  13340 s elapsed
Epoch 16,  1150/ 1501; acc:   0.00; ppl:   9.72; 3035 src tok/s; 4013 tgt tok/s;  13366 s elapsed
Epoch 16,  1200/ 1501; acc:   0.00; ppl:   9.99; 3335 src tok/s; 4299 tgt tok/s;  13394 s elapsed
Epoch 16,  1250/ 1501; acc:   0.00; ppl:   9.52; 2989 src tok/s; 3935 tgt tok/s;  13420 s elapsed
Epoch 16,  1300/ 1501; acc:   0.00; ppl:   9.90; 3121 src tok/s; 4085 tgt tok/s;  13447 s elapsed
Epoch 16,  1350/ 1501; acc:   0.00; ppl:  10.05; 2908 src tok/s; 3824 tgt tok/s;  13477 s elapsed
Epoch 16,  1400/ 1501; acc:   0.00; ppl:  11.14; 3331 src tok/s; 4201 tgt tok/s;  13507 s elapsed
Epoch 16,  1450/ 1501; acc:   0.00; ppl:  10.27; 3153 src tok/s; 4092 tgt tok/s;  13536 s elapsed
Epoch 16,  1500/ 1501; acc:   0.00; ppl:  10.00; 3294 src tok/s; 4289 tgt tok/s;  13562 s elapsed
Train perplexity: 10.2085
Train accuracy: 0
Validation perplexity: 19.041
Validation accuracy: 0
Decaying learning rate to 0.00195312

Epoch 17,    50/ 1501; acc:   0.00; ppl:   9.25; 3012 src tok/s; 3978 tgt tok/s;  13590 s elapsed
Epoch 17,   100/ 1501; acc:   0.00; ppl:  10.21; 3256 src tok/s; 4228 tgt tok/s;  13619 s elapsed
Epoch 17,   150/ 1501; acc:   0.00; ppl:  10.34; 3278 src tok/s; 4216 tgt tok/s;  13646 s elapsed
Epoch 17,   200/ 1501; acc:   0.00; ppl:   9.92; 3213 src tok/s; 4150 tgt tok/s;  13674 s elapsed
Epoch 17,   250/ 1501; acc:   0.00; ppl:  10.05; 2903 src tok/s; 3836 tgt tok/s;  13701 s elapsed
Epoch 17,   300/ 1501; acc:   0.00; ppl:  10.75; 3342 src tok/s; 4267 tgt tok/s;  13730 s elapsed
Epoch 17,   350/ 1501; acc:   0.00; ppl:  10.03; 3166 src tok/s; 4144 tgt tok/s;  13756 s elapsed
Epoch 17,   400/ 1501; acc:   0.00; ppl:  10.44; 3275 src tok/s; 4206 tgt tok/s;  13785 s elapsed
Epoch 17,   450/ 1501; acc:   0.00; ppl:   9.43; 3139 src tok/s; 4133 tgt tok/s;  13811 s elapsed
Epoch 17,   500/ 1501; acc:   0.00; ppl:  10.26; 3264 src tok/s; 4192 tgt tok/s;  13839 s elapsed
Epoch 17,   550/ 1501; acc:   0.00; ppl:  10.48; 3304 src tok/s; 4229 tgt tok/s;  13868 s elapsed
Epoch 17,   600/ 1501; acc:   0.00; ppl:  10.63; 3297 src tok/s; 4228 tgt tok/s;  13897 s elapsed
Epoch 17,   650/ 1501; acc:   0.00; ppl:  10.61; 3224 src tok/s; 4189 tgt tok/s;  13925 s elapsed
Epoch 17,   700/ 1501; acc:   0.00; ppl:  10.08; 3336 src tok/s; 4320 tgt tok/s;  13951 s elapsed
Epoch 17,   750/ 1501; acc:   0.00; ppl:   9.82; 3186 src tok/s; 4117 tgt tok/s;  13979 s elapsed
Epoch 17,   800/ 1501; acc:   0.00; ppl:   9.74; 3063 src tok/s; 4041 tgt tok/s;  14007 s elapsed
Epoch 17,   850/ 1501; acc:   0.00; ppl:  10.11; 3261 src tok/s; 4253 tgt tok/s;  14035 s elapsed
Epoch 17,   900/ 1501; acc:   0.00; ppl:  10.52; 3302 src tok/s; 4239 tgt tok/s;  14063 s elapsed
Epoch 17,   950/ 1501; acc:   0.00; ppl:   9.60; 3113 src tok/s; 4083 tgt tok/s;  14089 s elapsed
Epoch 17,  1000/ 1501; acc:   0.00; ppl:  10.23; 3271 src tok/s; 4203 tgt tok/s;  14117 s elapsed
Epoch 17,  1050/ 1501; acc:   0.00; ppl:  10.39; 3319 src tok/s; 4293 tgt tok/s;  14145 s elapsed
Epoch 17,  1100/ 1501; acc:   0.00; ppl:  10.88; 3319 src tok/s; 4240 tgt tok/s;  14175 s elapsed
Epoch 17,  1150/ 1501; acc:   0.00; ppl:   9.32; 3102 src tok/s; 4127 tgt tok/s;  14200 s elapsed
Epoch 17,  1200/ 1501; acc:   0.00; ppl:  10.10; 3208 src tok/s; 4134 tgt tok/s;  14229 s elapsed
Epoch 17,  1250/ 1501; acc:   0.00; ppl:   9.68; 3085 src tok/s; 4023 tgt tok/s;  14255 s elapsed
Epoch 17,  1300/ 1501; acc:   0.00; ppl:   9.77; 3087 src tok/s; 4004 tgt tok/s;  14282 s elapsed
Epoch 17,  1350/ 1501; acc:   0.00; ppl:  10.30; 3171 src tok/s; 4081 tgt tok/s;  14312 s elapsed
Epoch 17,  1400/ 1501; acc:   0.00; ppl:  10.79; 3342 src tok/s; 4265 tgt tok/s;  14341 s elapsed
Epoch 17,  1450/ 1501; acc:   0.00; ppl:  10.60; 3244 src tok/s; 4164 tgt tok/s;  14371 s elapsed
Epoch 17,  1500/ 1501; acc:   0.00; ppl:  10.75; 3320 src tok/s; 4275 tgt tok/s;  14401 s elapsed
Train perplexity: 10.1833
Train accuracy: 0
Validation perplexity: 19.0319
Validation accuracy: 0
Decaying learning rate to 0.000976562

Epoch 18,    50/ 1501; acc:   0.00; ppl:   8.80; 2943 src tok/s; 3938 tgt tok/s;  14427 s elapsed
Epoch 18,   100/ 1501; acc:   0.00; ppl:   9.80; 3245 src tok/s; 4223 tgt tok/s;  14454 s elapsed
Epoch 18,   150/ 1501; acc:   0.00; ppl:  10.16; 3136 src tok/s; 4034 tgt tok/s;  14484 s elapsed
Epoch 18,   200/ 1501; acc:   0.00; ppl:  10.01; 3023 src tok/s; 3910 tgt tok/s;  14513 s elapsed
Epoch 18,   250/ 1501; acc:   0.00; ppl:  10.38; 3112 src tok/s; 4037 tgt tok/s;  14542 s elapsed
Epoch 18,   300/ 1501; acc:   0.00; ppl:   9.33; 2983 src tok/s; 3932 tgt tok/s;  14569 s elapsed
Epoch 18,   350/ 1501; acc:   0.00; ppl:   9.75; 3108 src tok/s; 4084 tgt tok/s;  14595 s elapsed
Epoch 18,   400/ 1501; acc:   0.00; ppl:  10.60; 3311 src tok/s; 4235 tgt tok/s;  14624 s elapsed
Epoch 18,   450/ 1501; acc:   0.00; ppl:  10.72; 3370 src tok/s; 4240 tgt tok/s;  14653 s elapsed
Epoch 18,   500/ 1501; acc:   0.00; ppl:  10.06; 3149 src tok/s; 4107 tgt tok/s;  14680 s elapsed
Epoch 18,   550/ 1501; acc:   0.00; ppl:  10.13; 3155 src tok/s; 4067 tgt tok/s;  14708 s elapsed
Epoch 18,   600/ 1501; acc:   0.00; ppl:  10.49; 3278 src tok/s; 4250 tgt tok/s;  14736 s elapsed
Epoch 18,   650/ 1501; acc:   0.00; ppl:  10.75; 3156 src tok/s; 4050 tgt tok/s;  14768 s elapsed
Epoch 18,   700/ 1501; acc:   0.00; ppl:  10.83; 3266 src tok/s; 4180 tgt tok/s;  14798 s elapsed
Epoch 18,   750/ 1501; acc:   0.00; ppl:  10.45; 3132 src tok/s; 4038 tgt tok/s;  14828 s elapsed
Epoch 18,   800/ 1501; acc:   0.00; ppl:  10.01; 3123 src tok/s; 4033 tgt tok/s;  14858 s elapsed
Epoch 18,   850/ 1501; acc:   0.00; ppl:   9.54; 2865 src tok/s; 3782 tgt tok/s;  14886 s elapsed
Epoch 18,   900/ 1501; acc:   0.00; ppl:  10.65; 3253 src tok/s; 4156 tgt tok/s;  14916 s elapsed
Epoch 18,   950/ 1501; acc:   0.00; ppl:   9.40; 3122 src tok/s; 4116 tgt tok/s;  14940 s elapsed
Epoch 18,  1000/ 1501; acc:   0.00; ppl:  10.98; 3322 src tok/s; 4285 tgt tok/s;  14971 s elapsed
Epoch 18,  1050/ 1501; acc:   0.00; ppl:   9.93; 3217 src tok/s; 4204 tgt tok/s;  14998 s elapsed
Epoch 18,  1100/ 1501; acc:   0.00; ppl:   9.98; 3284 src tok/s; 4226 tgt tok/s;  15024 s elapsed
Epoch 18,  1150/ 1501; acc:   0.00; ppl:  10.68; 3214 src tok/s; 4171 tgt tok/s;  15056 s elapsed
Epoch 18,  1200/ 1501; acc:   0.00; ppl:   9.90; 3247 src tok/s; 4203 tgt tok/s;  15084 s elapsed
Epoch 18,  1250/ 1501; acc:   0.00; ppl:   9.72; 3144 src tok/s; 4134 tgt tok/s;  15111 s elapsed
Epoch 18,  1300/ 1501; acc:   0.00; ppl:   9.87; 3123 src tok/s; 4120 tgt tok/s;  15136 s elapsed
Epoch 18,  1350/ 1501; acc:   0.00; ppl:  10.93; 3324 src tok/s; 4233 tgt tok/s;  15166 s elapsed
Epoch 18,  1400/ 1501; acc:   0.00; ppl:  10.17; 3303 src tok/s; 4259 tgt tok/s;  15193 s elapsed
Epoch 18,  1450/ 1501; acc:   0.00; ppl:  10.86; 3207 src tok/s; 4056 tgt tok/s;  15223 s elapsed
Epoch 18,  1500/ 1501; acc:   0.00; ppl:   9.71; 3121 src tok/s; 4086 tgt tok/s;  15249 s elapsed
Train perplexity: 10.1783
Train accuracy: 0
Validation perplexity: 19.0292
Validation accuracy: 0
Decaying learning rate to 0.000488281

Epoch 19,    50/ 1501; acc:   0.00; ppl:  10.10; 3231 src tok/s; 4167 tgt tok/s;  15278 s elapsed
Epoch 19,   100/ 1501; acc:   0.00; ppl:  10.02; 3226 src tok/s; 4135 tgt tok/s;  15307 s elapsed
Epoch 19,   150/ 1501; acc:   0.00; ppl:  10.12; 3144 src tok/s; 4091 tgt tok/s;  15337 s elapsed
Epoch 19,   200/ 1501; acc:   0.00; ppl:  11.03; 3395 src tok/s; 4265 tgt tok/s;  15369 s elapsed
Epoch 19,   250/ 1501; acc:   0.00; ppl:   9.67; 3134 src tok/s; 4125 tgt tok/s;  15397 s elapsed
Epoch 19,   300/ 1501; acc:   0.00; ppl:   9.55; 3103 src tok/s; 4097 tgt tok/s;  15423 s elapsed
Epoch 19,   350/ 1501; acc:   0.00; ppl:   9.59; 2917 src tok/s; 3835 tgt tok/s;  15450 s elapsed
Epoch 19,   400/ 1501; acc:   0.00; ppl:  10.41; 3058 src tok/s; 3925 tgt tok/s;  15482 s elapsed
Epoch 19,   450/ 1501; acc:   0.00; ppl:  10.45; 3176 src tok/s; 4078 tgt tok/s;  15511 s elapsed
Epoch 19,   500/ 1501; acc:   0.00; ppl:  10.34; 3088 src tok/s; 4004 tgt tok/s;  15541 s elapsed
Epoch 19,   550/ 1501; acc:   0.00; ppl:  10.88; 3324 src tok/s; 4225 tgt tok/s;  15570 s elapsed
Epoch 19,   600/ 1501; acc:   0.00; ppl:   9.87; 3295 src tok/s; 4297 tgt tok/s;  15597 s elapsed
Epoch 19,   650/ 1501; acc:   0.00; ppl:   9.99; 3258 src tok/s; 4233 tgt tok/s;  15622 s elapsed
Epoch 19,   700/ 1501; acc:   0.00; ppl:   9.47; 2939 src tok/s; 3916 tgt tok/s;  15648 s elapsed
Epoch 19,   750/ 1501; acc:   0.00; ppl:  10.26; 3084 src tok/s; 4012 tgt tok/s;  15678 s elapsed
Epoch 19,   800/ 1501; acc:   0.00; ppl:  11.02; 3372 src tok/s; 4280 tgt tok/s;  15708 s elapsed
Epoch 19,   850/ 1501; acc:   0.00; ppl:  11.08; 2240 src tok/s; 2828 tgt tok/s;  15755 s elapsed
Epoch 19,   900/ 1501; acc:   0.00; ppl:  10.13; 2431 src tok/s; 3171 tgt tok/s;  15791 s elapsed
Epoch 19,   950/ 1501; acc:   0.00; ppl:  10.91; 3393 src tok/s; 4283 tgt tok/s;  15821 s elapsed
Epoch 19,  1000/ 1501; acc:   0.00; ppl:   9.98; 3220 src tok/s; 4193 tgt tok/s;  15847 s elapsed
Epoch 19,  1050/ 1501; acc:   0.00; ppl:  10.09; 3304 src tok/s; 4285 tgt tok/s;  15873 s elapsed
Epoch 19,  1100/ 1501; acc:   0.00; ppl:   9.65; 2177 src tok/s; 2897 tgt tok/s;  15910 s elapsed
Epoch 19,  1150/ 1501; acc:   0.00; ppl:  10.32; 2195 src tok/s; 2824 tgt tok/s;  15953 s elapsed
Epoch 19,  1200/ 1501; acc:   0.00; ppl:   9.78; 3143 src tok/s; 4104 tgt tok/s;  15979 s elapsed
Epoch 19,  1250/ 1501; acc:   0.00; ppl:  10.58; 3313 src tok/s; 4242 tgt tok/s;  16008 s elapsed
Epoch 19,  1300/ 1501; acc:   0.00; ppl:  10.03; 3214 src tok/s; 4184 tgt tok/s;  16036 s elapsed
Epoch 19,  1350/ 1501; acc:   0.00; ppl:   9.67; 3021 src tok/s; 4004 tgt tok/s;  16062 s elapsed
Epoch 19,  1400/ 1501; acc:   0.00; ppl:   9.87; 3214 src tok/s; 4233 tgt tok/s;  16090 s elapsed
Epoch 19,  1450/ 1501; acc:   0.00; ppl:  10.17; 3191 src tok/s; 4107 tgt tok/s;  16116 s elapsed
Epoch 19,  1500/ 1501; acc:   0.00; ppl:   9.71; 3136 src tok/s; 4083 tgt tok/s;  16143 s elapsed
Train perplexity: 10.1775
Train accuracy: 0
Validation perplexity: 19.0263
Validation accuracy: 0
Decaying learning rate to 0.000244141

Epoch 20,    50/ 1501; acc:   0.00; ppl:  10.16; 3204 src tok/s; 4076 tgt tok/s;  16173 s elapsed
Epoch 20,   100/ 1501; acc:   0.00; ppl:  10.83; 3405 src tok/s; 4323 tgt tok/s;  16202 s elapsed
Epoch 20,   150/ 1501; acc:   0.00; ppl:   9.91; 3177 src tok/s; 4151 tgt tok/s;  16229 s elapsed
Epoch 20,   200/ 1501; acc:   0.00; ppl:  11.18; 3359 src tok/s; 4186 tgt tok/s;  16258 s elapsed
Epoch 20,   250/ 1501; acc:   0.00; ppl:   9.30; 3004 src tok/s; 3954 tgt tok/s;  16284 s elapsed
Epoch 20,   300/ 1501; acc:   0.00; ppl:  10.63; 3397 src tok/s; 4316 tgt tok/s;  16311 s elapsed
Epoch 20,   350/ 1501; acc:   0.00; ppl:  10.42; 3268 src tok/s; 4209 tgt tok/s;  16340 s elapsed
Epoch 20,   400/ 1501; acc:   0.00; ppl:  10.26; 3174 src tok/s; 4151 tgt tok/s;  16368 s elapsed
Epoch 20,   450/ 1501; acc:   0.00; ppl:   9.95; 3193 src tok/s; 4162 tgt tok/s;  16396 s elapsed
Epoch 20,   500/ 1501; acc:   0.00; ppl:   9.60; 3158 src tok/s; 4152 tgt tok/s;  16422 s elapsed
Epoch 20,   550/ 1501; acc:   0.00; ppl:  10.21; 3203 src tok/s; 4182 tgt tok/s;  16450 s elapsed
Epoch 20,   600/ 1501; acc:   0.00; ppl:  10.11; 3194 src tok/s; 4101 tgt tok/s;  16478 s elapsed
Epoch 20,   650/ 1501; acc:   0.00; ppl:   9.41; 3134 src tok/s; 4148 tgt tok/s;  16504 s elapsed
Epoch 20,   700/ 1501; acc:   0.00; ppl:  10.18; 3188 src tok/s; 4130 tgt tok/s;  16532 s elapsed
Epoch 20,   750/ 1501; acc:   0.00; ppl:   9.85; 3187 src tok/s; 4191 tgt tok/s;  16558 s elapsed
Epoch 20,   800/ 1501; acc:   0.00; ppl:  10.04; 3176 src tok/s; 4121 tgt tok/s;  16585 s elapsed
Epoch 20,   850/ 1501; acc:   0.00; ppl:  10.50; 3265 src tok/s; 4177 tgt tok/s;  16615 s elapsed
Epoch 20,   900/ 1501; acc:   0.00; ppl:  10.37; 3188 src tok/s; 4102 tgt tok/s;  16645 s elapsed
Epoch 20,   950/ 1501; acc:   0.00; ppl:  10.29; 3254 src tok/s; 4213 tgt tok/s;  16674 s elapsed
Epoch 20,  1000/ 1501; acc:   0.00; ppl:   9.39; 3069 src tok/s; 4094 tgt tok/s;  16700 s elapsed
Epoch 20,  1050/ 1501; acc:   0.00; ppl:   9.89; 3175 src tok/s; 4133 tgt tok/s;  16728 s elapsed
Epoch 20,  1100/ 1501; acc:   0.00; ppl:  10.28; 3225 src tok/s; 4197 tgt tok/s;  16755 s elapsed
Epoch 20,  1150/ 1501; acc:   0.00; ppl:  10.72; 3347 src tok/s; 4246 tgt tok/s;  16784 s elapsed
Epoch 20,  1200/ 1501; acc:   0.00; ppl:  10.32; 3291 src tok/s; 4253 tgt tok/s;  16813 s elapsed
Epoch 20,  1250/ 1501; acc:   0.00; ppl:  10.48; 3222 src tok/s; 4144 tgt tok/s;  16842 s elapsed
Epoch 20,  1300/ 1501; acc:   0.00; ppl:  10.34; 3066 src tok/s; 4018 tgt tok/s;  16870 s elapsed
Epoch 20,  1350/ 1501; acc:   0.00; ppl:  10.09; 3210 src tok/s; 4175 tgt tok/s;  16898 s elapsed
Epoch 20,  1400/ 1501; acc:   0.00; ppl:   9.72; 3254 src tok/s; 4232 tgt tok/s;  16924 s elapsed
Epoch 20,  1450/ 1501; acc:   0.00; ppl:  10.05; 3274 src tok/s; 4249 tgt tok/s;  16950 s elapsed
Epoch 20,  1500/ 1501; acc:   0.00; ppl:  10.50; 3214 src tok/s; 4165 tgt tok/s;  16981 s elapsed
Train perplexity: 10.1753
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 0.00012207

Epoch 21,    50/ 1501; acc:   0.00; ppl:  10.53; 3348 src tok/s; 4268 tgt tok/s;  17011 s elapsed
Epoch 21,   100/ 1501; acc:   0.00; ppl:   9.78; 3101 src tok/s; 4069 tgt tok/s;  17037 s elapsed
Epoch 21,   150/ 1501; acc:   0.00; ppl:  10.52; 3226 src tok/s; 4192 tgt tok/s;  17067 s elapsed
Epoch 21,   200/ 1501; acc:   0.00; ppl:  10.31; 3278 src tok/s; 4198 tgt tok/s;  17094 s elapsed
Epoch 21,   250/ 1501; acc:   0.00; ppl:  10.80; 3136 src tok/s; 4082 tgt tok/s;  17122 s elapsed
Epoch 21,   300/ 1501; acc:   0.00; ppl:  10.30; 3248 src tok/s; 4203 tgt tok/s;  17151 s elapsed
Epoch 21,   350/ 1501; acc:   0.00; ppl:  11.05; 3398 src tok/s; 4308 tgt tok/s;  17182 s elapsed
Epoch 21,   400/ 1501; acc:   0.00; ppl:   9.81; 3276 src tok/s; 4293 tgt tok/s;  17209 s elapsed
Epoch 21,   450/ 1501; acc:   0.00; ppl:   9.67; 3136 src tok/s; 4101 tgt tok/s;  17236 s elapsed
Epoch 21,   500/ 1501; acc:   0.00; ppl:   9.99; 3130 src tok/s; 4095 tgt tok/s;  17264 s elapsed
Epoch 21,   550/ 1501; acc:   0.00; ppl:  10.36; 3003 src tok/s; 3872 tgt tok/s;  17292 s elapsed
Epoch 21,   600/ 1501; acc:   0.00; ppl:   9.97; 3139 src tok/s; 4074 tgt tok/s;  17319 s elapsed
Epoch 21,   650/ 1501; acc:   0.00; ppl:   9.94; 3189 src tok/s; 4169 tgt tok/s;  17347 s elapsed
Epoch 21,   700/ 1501; acc:   0.00; ppl:   9.29; 2972 src tok/s; 3904 tgt tok/s;  17374 s elapsed
Epoch 21,   750/ 1501; acc:   0.00; ppl:   9.97; 3221 src tok/s; 4175 tgt tok/s;  17402 s elapsed
Epoch 21,   800/ 1501; acc:   0.00; ppl:   9.70; 3142 src tok/s; 4106 tgt tok/s;  17429 s elapsed
Epoch 21,   850/ 1501; acc:   0.00; ppl:  10.95; 3226 src tok/s; 4096 tgt tok/s;  17458 s elapsed
Epoch 21,   900/ 1501; acc:   0.00; ppl:   9.94; 3161 src tok/s; 4128 tgt tok/s;  17486 s elapsed
Epoch 21,   950/ 1501; acc:   0.00; ppl:  10.43; 3221 src tok/s; 4175 tgt tok/s;  17516 s elapsed
Epoch 21,  1000/ 1501; acc:   0.00; ppl:  10.64; 3285 src tok/s; 4192 tgt tok/s;  17547 s elapsed
Epoch 21,  1050/ 1501; acc:   0.00; ppl:   9.69; 3118 src tok/s; 4096 tgt tok/s;  17574 s elapsed
Epoch 21,  1100/ 1501; acc:   0.00; ppl:  10.38; 3293 src tok/s; 4204 tgt tok/s;  17602 s elapsed
Epoch 21,  1150/ 1501; acc:   0.00; ppl:  10.56; 3360 src tok/s; 4315 tgt tok/s;  17631 s elapsed
Epoch 21,  1200/ 1501; acc:   0.00; ppl:   9.93; 3171 src tok/s; 4117 tgt tok/s;  17658 s elapsed
Epoch 21,  1250/ 1501; acc:   0.00; ppl:   9.71; 3238 src tok/s; 4216 tgt tok/s;  17683 s elapsed
Epoch 21,  1300/ 1501; acc:   0.00; ppl:  10.69; 3300 src tok/s; 4232 tgt tok/s;  17713 s elapsed
Epoch 21,  1350/ 1501; acc:   0.00; ppl:  10.07; 3257 src tok/s; 4196 tgt tok/s;  17741 s elapsed
Epoch 21,  1400/ 1501; acc:   0.00; ppl:  10.38; 3384 src tok/s; 4318 tgt tok/s;  17768 s elapsed
Epoch 21,  1450/ 1501; acc:   0.00; ppl:   9.40; 3033 src tok/s; 4027 tgt tok/s;  17794 s elapsed
Epoch 21,  1500/ 1501; acc:   0.00; ppl:  10.20; 3176 src tok/s; 4133 tgt tok/s;  17822 s elapsed
Train perplexity: 10.1775
Train accuracy: 0
Validation perplexity: 19.0266
Validation accuracy: 0
Decaying learning rate to 6.10352e-05

Epoch 22,    50/ 1501; acc:   0.00; ppl:   9.71; 3101 src tok/s; 4062 tgt tok/s;  17850 s elapsed
Epoch 22,   100/ 1501; acc:   0.00; ppl:  10.66; 3243 src tok/s; 4163 tgt tok/s;  17880 s elapsed
Epoch 22,   150/ 1501; acc:   0.00; ppl:  10.08; 3275 src tok/s; 4213 tgt tok/s;  17908 s elapsed
Epoch 22,   200/ 1501; acc:   0.00; ppl:  10.96; 3359 src tok/s; 4243 tgt tok/s;  17938 s elapsed
Epoch 22,   250/ 1501; acc:   0.00; ppl:  10.18; 3227 src tok/s; 4177 tgt tok/s;  17966 s elapsed
Epoch 22,   300/ 1501; acc:   0.00; ppl:  10.10; 3298 src tok/s; 4300 tgt tok/s;  17994 s elapsed
Epoch 22,   350/ 1501; acc:   0.00; ppl:  10.58; 3203 src tok/s; 4158 tgt tok/s;  18024 s elapsed
Epoch 22,   400/ 1501; acc:   0.00; ppl:  10.40; 3159 src tok/s; 4050 tgt tok/s;  18052 s elapsed
Epoch 22,   450/ 1501; acc:   0.00; ppl:  10.09; 3201 src tok/s; 4123 tgt tok/s;  18079 s elapsed
Epoch 22,   500/ 1501; acc:   0.00; ppl:   9.48; 3092 src tok/s; 4083 tgt tok/s;  18105 s elapsed
Epoch 22,   550/ 1501; acc:   0.00; ppl:  10.15; 3352 src tok/s; 4340 tgt tok/s;  18133 s elapsed
Epoch 22,   600/ 1501; acc:   0.00; ppl:  10.09; 3029 src tok/s; 3930 tgt tok/s;  18163 s elapsed
Epoch 22,   650/ 1501; acc:   0.00; ppl:  11.15; 3342 src tok/s; 4257 tgt tok/s;  18193 s elapsed
Epoch 22,   700/ 1501; acc:   0.00; ppl:   9.91; 3163 src tok/s; 4137 tgt tok/s;  18220 s elapsed
Epoch 22,   750/ 1501; acc:   0.00; ppl:   9.57; 3086 src tok/s; 4022 tgt tok/s;  18246 s elapsed
Epoch 22,   800/ 1501; acc:   0.00; ppl:  10.65; 3152 src tok/s; 4031 tgt tok/s;  18277 s elapsed
Epoch 22,   850/ 1501; acc:   0.00; ppl:   9.90; 2958 src tok/s; 3880 tgt tok/s;  18306 s elapsed
Epoch 22,   900/ 1501; acc:   0.00; ppl:   9.72; 3223 src tok/s; 4186 tgt tok/s;  18332 s elapsed
Epoch 22,   950/ 1501; acc:   0.00; ppl:   9.83; 3197 src tok/s; 4177 tgt tok/s;  18359 s elapsed
Epoch 22,  1000/ 1501; acc:   0.00; ppl:   9.97; 3260 src tok/s; 4223 tgt tok/s;  18387 s elapsed
Epoch 22,  1050/ 1501; acc:   0.00; ppl:   9.94; 3019 src tok/s; 3941 tgt tok/s;  18415 s elapsed
Epoch 22,  1100/ 1501; acc:   0.00; ppl:  10.19; 3161 src tok/s; 4081 tgt tok/s;  18444 s elapsed
Epoch 22,  1150/ 1501; acc:   0.00; ppl:  10.21; 3309 src tok/s; 4262 tgt tok/s;  18471 s elapsed
Epoch 22,  1200/ 1501; acc:   0.00; ppl:   9.67; 3057 src tok/s; 4026 tgt tok/s;  18499 s elapsed
Epoch 22,  1250/ 1501; acc:   0.00; ppl:   9.59; 3201 src tok/s; 4171 tgt tok/s;  18523 s elapsed
Epoch 22,  1300/ 1501; acc:   0.00; ppl:  10.14; 3360 src tok/s; 4333 tgt tok/s;  18550 s elapsed
Epoch 22,  1350/ 1501; acc:   0.00; ppl:  11.10; 3271 src tok/s; 4161 tgt tok/s;  18582 s elapsed
Epoch 22,  1400/ 1501; acc:   0.00; ppl:   9.80; 3196 src tok/s; 4195 tgt tok/s;  18608 s elapsed
Epoch 22,  1450/ 1501; acc:   0.00; ppl:  10.52; 3316 src tok/s; 4287 tgt tok/s;  18637 s elapsed
Epoch 22,  1500/ 1501; acc:   0.00; ppl:  10.40; 3288 src tok/s; 4256 tgt tok/s;  18664 s elapsed
Train perplexity: 10.1715
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 3.05176e-05

Epoch 23,    50/ 1501; acc:   0.00; ppl:  10.76; 3183 src tok/s; 4065 tgt tok/s;  18696 s elapsed
Epoch 23,   100/ 1501; acc:   0.00; ppl:   9.55; 3057 src tok/s; 4033 tgt tok/s;  18723 s elapsed
Epoch 23,   150/ 1501; acc:   0.00; ppl:  10.01; 3292 src tok/s; 4255 tgt tok/s;  18751 s elapsed
Epoch 23,   200/ 1501; acc:   0.00; ppl:  10.03; 3088 src tok/s; 4041 tgt tok/s;  18779 s elapsed
Epoch 23,   250/ 1501; acc:   0.00; ppl:   9.91; 3115 src tok/s; 4085 tgt tok/s;  18805 s elapsed
Epoch 23,   300/ 1501; acc:   0.00; ppl:  10.80; 3362 src tok/s; 4333 tgt tok/s;  18835 s elapsed
Epoch 23,   350/ 1501; acc:   0.00; ppl:  10.40; 3377 src tok/s; 4355 tgt tok/s;  18864 s elapsed
Epoch 23,   400/ 1501; acc:   0.00; ppl:  10.11; 3130 src tok/s; 4086 tgt tok/s;  18892 s elapsed
Epoch 23,   450/ 1501; acc:   0.00; ppl:  10.01; 3268 src tok/s; 4244 tgt tok/s;  18918 s elapsed
Epoch 23,   500/ 1501; acc:   0.00; ppl:  10.18; 3342 src tok/s; 4285 tgt tok/s;  18944 s elapsed
Epoch 23,   550/ 1501; acc:   0.00; ppl:  10.37; 3285 src tok/s; 4252 tgt tok/s;  18972 s elapsed
Epoch 23,   600/ 1501; acc:   0.00; ppl:  10.19; 3244 src tok/s; 4205 tgt tok/s;  19001 s elapsed
Epoch 23,   650/ 1501; acc:   0.00; ppl:  10.59; 3343 src tok/s; 4273 tgt tok/s;  19030 s elapsed
Epoch 23,   700/ 1501; acc:   0.00; ppl:   9.69; 3218 src tok/s; 4222 tgt tok/s;  19056 s elapsed
Epoch 23,   750/ 1501; acc:   0.00; ppl:   9.64; 3189 src tok/s; 4173 tgt tok/s;  19082 s elapsed
Epoch 23,   800/ 1501; acc:   0.00; ppl:  10.45; 3317 src tok/s; 4294 tgt tok/s;  19111 s elapsed
Epoch 23,   850/ 1501; acc:   0.00; ppl:   9.80; 3088 src tok/s; 4050 tgt tok/s;  19139 s elapsed
Epoch 23,   900/ 1501; acc:   0.00; ppl:   9.73; 3185 src tok/s; 4141 tgt tok/s;  19163 s elapsed
Epoch 23,   950/ 1501; acc:   0.00; ppl:   9.74; 3235 src tok/s; 4242 tgt tok/s;  19190 s elapsed
Epoch 23,  1000/ 1501; acc:   0.00; ppl:  10.27; 3370 src tok/s; 4343 tgt tok/s;  19217 s elapsed
Epoch 23,  1050/ 1501; acc:   0.00; ppl:  10.52; 3227 src tok/s; 4155 tgt tok/s;  19248 s elapsed
Epoch 23,  1100/ 1501; acc:   0.00; ppl:  10.06; 3274 src tok/s; 4203 tgt tok/s;  19276 s elapsed
Epoch 23,  1150/ 1501; acc:   0.00; ppl:   9.78; 2922 src tok/s; 3834 tgt tok/s;  19305 s elapsed
Epoch 23,  1200/ 1501; acc:   0.00; ppl:  10.13; 3186 src tok/s; 4057 tgt tok/s;  19331 s elapsed
Epoch 23,  1250/ 1501; acc:   0.00; ppl:   9.74; 3263 src tok/s; 4206 tgt tok/s;  19357 s elapsed
Epoch 23,  1300/ 1501; acc:   0.00; ppl:  10.75; 3306 src tok/s; 4290 tgt tok/s;  19386 s elapsed
Epoch 23,  1350/ 1501; acc:   0.00; ppl:  10.05; 3331 src tok/s; 4294 tgt tok/s;  19412 s elapsed
Epoch 23,  1400/ 1501; acc:   0.00; ppl:  10.28; 3352 src tok/s; 4324 tgt tok/s;  19440 s elapsed
Epoch 23,  1450/ 1501; acc:   0.00; ppl:  10.79; 3250 src tok/s; 4161 tgt tok/s;  19471 s elapsed
Epoch 23,  1500/ 1501; acc:   0.00; ppl:  10.58; 3224 src tok/s; 4137 tgt tok/s;  19498 s elapsed
Train perplexity: 10.1732
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 1.52588e-05

Epoch 24,    50/ 1501; acc:   0.00; ppl:  10.92; 3423 src tok/s; 4330 tgt tok/s;  19529 s elapsed
Epoch 24,   100/ 1501; acc:   0.00; ppl:  10.22; 3290 src tok/s; 4232 tgt tok/s;  19558 s elapsed
Epoch 24,   150/ 1501; acc:   0.00; ppl:  11.14; 3476 src tok/s; 4375 tgt tok/s;  19588 s elapsed
Epoch 24,   200/ 1501; acc:   0.00; ppl:   9.69; 3161 src tok/s; 4148 tgt tok/s;  19615 s elapsed
Epoch 24,   250/ 1501; acc:   0.00; ppl:  10.26; 3233 src tok/s; 4185 tgt tok/s;  19644 s elapsed
Epoch 24,   300/ 1501; acc:   0.00; ppl:  10.26; 3062 src tok/s; 3967 tgt tok/s;  19673 s elapsed
Epoch 24,   350/ 1501; acc:   0.00; ppl:  10.25; 3216 src tok/s; 4162 tgt tok/s;  19701 s elapsed
Epoch 24,   400/ 1501; acc:   0.00; ppl:  10.11; 3242 src tok/s; 4182 tgt tok/s;  19728 s elapsed
Epoch 24,   450/ 1501; acc:   0.00; ppl:  10.18; 3325 src tok/s; 4285 tgt tok/s;  19754 s elapsed
Epoch 24,   500/ 1501; acc:   0.00; ppl:  10.47; 3264 src tok/s; 4191 tgt tok/s;  19782 s elapsed
Epoch 24,   550/ 1501; acc:   0.00; ppl:   9.57; 3108 src tok/s; 4083 tgt tok/s;  19810 s elapsed
Epoch 24,   600/ 1501; acc:   0.00; ppl:   9.36; 2968 src tok/s; 3905 tgt tok/s;  19836 s elapsed
Epoch 24,   650/ 1501; acc:   0.00; ppl:  10.08; 3138 src tok/s; 4086 tgt tok/s;  19865 s elapsed
Epoch 24,   700/ 1501; acc:   0.00; ppl:  10.27; 3215 src tok/s; 4126 tgt tok/s;  19894 s elapsed
Epoch 24,   750/ 1501; acc:   0.00; ppl:  10.54; 3207 src tok/s; 4167 tgt tok/s;  19921 s elapsed
Epoch 24,   800/ 1501; acc:   0.00; ppl:  10.39; 3249 src tok/s; 4161 tgt tok/s;  19949 s elapsed
Epoch 24,   850/ 1501; acc:   0.00; ppl:   8.91; 2978 src tok/s; 3960 tgt tok/s;  19973 s elapsed
Epoch 24,   900/ 1501; acc:   0.00; ppl:  10.27; 3161 src tok/s; 4123 tgt tok/s;  20000 s elapsed
Epoch 24,   950/ 1501; acc:   0.00; ppl:   9.52; 3058 src tok/s; 4053 tgt tok/s;  20025 s elapsed
Epoch 24,  1000/ 1501; acc:   0.00; ppl:   9.81; 3227 src tok/s; 4199 tgt tok/s;  20052 s elapsed
Epoch 24,  1050/ 1501; acc:   0.00; ppl:  10.19; 3134 src tok/s; 4092 tgt tok/s;  20082 s elapsed
Epoch 24,  1100/ 1501; acc:   0.00; ppl:  11.15; 3437 src tok/s; 4376 tgt tok/s;  20112 s elapsed
Epoch 24,  1150/ 1501; acc:   0.00; ppl:  10.50; 3220 src tok/s; 4157 tgt tok/s;  20142 s elapsed
Epoch 24,  1200/ 1501; acc:   0.00; ppl:   9.63; 3187 src tok/s; 4166 tgt tok/s;  20169 s elapsed
Epoch 24,  1250/ 1501; acc:   0.00; ppl:  10.26; 3244 src tok/s; 4195 tgt tok/s;  20197 s elapsed
Epoch 24,  1300/ 1501; acc:   0.00; ppl:  10.03; 3209 src tok/s; 4171 tgt tok/s;  20224 s elapsed
Epoch 24,  1350/ 1501; acc:   0.00; ppl:  10.38; 3238 src tok/s; 4161 tgt tok/s;  20253 s elapsed
Epoch 24,  1400/ 1501; acc:   0.00; ppl:  10.23; 3267 src tok/s; 4233 tgt tok/s;  20281 s elapsed
Epoch 24,  1450/ 1501; acc:   0.00; ppl:   9.60; 3197 src tok/s; 4159 tgt tok/s;  20308 s elapsed
Epoch 24,  1500/ 1501; acc:   0.00; ppl:  10.29; 3208 src tok/s; 4172 tgt tok/s;  20337 s elapsed
Train perplexity: 10.1686
Train accuracy: 0
Validation perplexity: 19.0264
Validation accuracy: 0
Decaying learning rate to 7.62939e-06

Epoch 25,    50/ 1501; acc:   0.00; ppl:   9.89; 3207 src tok/s; 4170 tgt tok/s;  20366 s elapsed
Epoch 25,   100/ 1501; acc:   0.00; ppl:   9.89; 3249 src tok/s; 4233 tgt tok/s;  20393 s elapsed
Epoch 25,   150/ 1501; acc:   0.00; ppl:  10.53; 3478 src tok/s; 4398 tgt tok/s;  20420 s elapsed
Epoch 25,   200/ 1501; acc:   0.00; ppl:   9.60; 3088 src tok/s; 4062 tgt tok/s;  20446 s elapsed
Epoch 25,   250/ 1501; acc:   0.00; ppl:   9.88; 3231 src tok/s; 4238 tgt tok/s;  20470 s elapsed
Epoch 25,   300/ 1501; acc:   0.00; ppl:   9.88; 3015 src tok/s; 3943 tgt tok/s;  20499 s elapsed
Epoch 25,   350/ 1501; acc:   0.00; ppl:  10.30; 3290 src tok/s; 4233 tgt tok/s;  20528 s elapsed
Epoch 25,   400/ 1501; acc:   0.00; ppl:  10.45; 3170 src tok/s; 4052 tgt tok/s;  20558 s elapsed
Epoch 25,   450/ 1501; acc:   0.00; ppl:  10.14; 3001 src tok/s; 3882 tgt tok/s;  20588 s elapsed
Epoch 25,   500/ 1501; acc:   0.00; ppl:  10.02; 2987 src tok/s; 3891 tgt tok/s;  20619 s elapsed
Epoch 25,   550/ 1501; acc:   0.00; ppl:  10.48; 3013 src tok/s; 3910 tgt tok/s;  20649 s elapsed
Epoch 25,   600/ 1501; acc:   0.00; ppl:  10.41; 3060 src tok/s; 3967 tgt tok/s;  20679 s elapsed
Epoch 25,   650/ 1501; acc:   0.00; ppl:   9.99; 2978 src tok/s; 3866 tgt tok/s;  20706 s elapsed
Epoch 25,   700/ 1501; acc:   0.00; ppl:  10.40; 3283 src tok/s; 4198 tgt tok/s;  20734 s elapsed
Epoch 25,   750/ 1501; acc:   0.00; ppl:  10.28; 3065 src tok/s; 3936 tgt tok/s;  20763 s elapsed
Epoch 25,   800/ 1501; acc:   0.00; ppl:  10.39; 3122 src tok/s; 3999 tgt tok/s;  20794 s elapsed
Epoch 25,   850/ 1501; acc:   0.00; ppl:  10.43; 3031 src tok/s; 3923 tgt tok/s;  20826 s elapsed
Epoch 25,   900/ 1501; acc:   0.00; ppl:  10.12; 3198 src tok/s; 4167 tgt tok/s;  20855 s elapsed
Epoch 25,   950/ 1501; acc:   0.00; ppl:  11.15; 3242 src tok/s; 4133 tgt tok/s;  20886 s elapsed
Epoch 25,  1000/ 1501; acc:   0.00; ppl:  11.05; 3360 src tok/s; 4221 tgt tok/s;  20918 s elapsed
Epoch 25,  1050/ 1501; acc:   0.00; ppl:  10.11; 3115 src tok/s; 4074 tgt tok/s;  20946 s elapsed
Epoch 25,  1100/ 1501; acc:   0.00; ppl:  10.39; 3156 src tok/s; 4059 tgt tok/s;  20974 s elapsed
Epoch 25,  1150/ 1501; acc:   0.00; ppl:   9.50; 2983 src tok/s; 3930 tgt tok/s;  21002 s elapsed
Epoch 25,  1200/ 1501; acc:   0.00; ppl:   9.59; 2901 src tok/s; 3830 tgt tok/s;  21030 s elapsed
Epoch 25,  1250/ 1501; acc:   0.00; ppl:   9.68; 3161 src tok/s; 4168 tgt tok/s;  21057 s elapsed
Epoch 25,  1300/ 1501; acc:   0.00; ppl:   9.88; 3118 src tok/s; 4079 tgt tok/s;  21083 s elapsed
Epoch 25,  1350/ 1501; acc:   0.00; ppl:  10.38; 3173 src tok/s; 4087 tgt tok/s;  21112 s elapsed
Epoch 25,  1400/ 1501; acc:   0.00; ppl:   9.89; 3128 src tok/s; 4067 tgt tok/s;  21140 s elapsed
Epoch 25,  1450/ 1501; acc:   0.00; ppl:  10.26; 3189 src tok/s; 4142 tgt tok/s;  21169 s elapsed
Epoch 25,  1500/ 1501; acc:   0.00; ppl:   9.90; 3233 src tok/s; 4181 tgt tok/s;  21196 s elapsed
Train perplexity: 10.1721
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 3.8147e-06

Epoch 26,    50/ 1501; acc:   0.00; ppl:  10.07; 3105 src tok/s; 4023 tgt tok/s;  21227 s elapsed
Epoch 26,   100/ 1501; acc:   0.00; ppl:   9.72; 3155 src tok/s; 4111 tgt tok/s;  21254 s elapsed
Epoch 26,   150/ 1501; acc:   0.00; ppl:  10.27; 3328 src tok/s; 4271 tgt tok/s;  21283 s elapsed
Epoch 26,   200/ 1501; acc:   0.00; ppl:   9.98; 3232 src tok/s; 4185 tgt tok/s;  21309 s elapsed
Epoch 26,   250/ 1501; acc:   0.00; ppl:  11.22; 3453 src tok/s; 4361 tgt tok/s;  21339 s elapsed
Epoch 26,   300/ 1501; acc:   0.00; ppl:  10.19; 3266 src tok/s; 4254 tgt tok/s;  21367 s elapsed
Epoch 26,   350/ 1501; acc:   0.00; ppl:   9.96; 3145 src tok/s; 4100 tgt tok/s;  21396 s elapsed
Epoch 26,   400/ 1501; acc:   0.00; ppl:   9.92; 3087 src tok/s; 4056 tgt tok/s;  21424 s elapsed
Epoch 26,   450/ 1501; acc:   0.00; ppl:   9.90; 3209 src tok/s; 4152 tgt tok/s;  21451 s elapsed
Epoch 26,   500/ 1501; acc:   0.00; ppl:   9.08; 2787 src tok/s; 3692 tgt tok/s;  21478 s elapsed
Epoch 26,   550/ 1501; acc:   0.00; ppl:  10.17; 3205 src tok/s; 4127 tgt tok/s;  21506 s elapsed
Epoch 26,   600/ 1501; acc:   0.00; ppl:   9.28; 2881 src tok/s; 3819 tgt tok/s;  21532 s elapsed
Epoch 26,   650/ 1501; acc:   0.00; ppl:  11.09; 2991 src tok/s; 3791 tgt tok/s;  21564 s elapsed
Epoch 26,   700/ 1501; acc:   0.00; ppl:   9.70; 2888 src tok/s; 3781 tgt tok/s;  21593 s elapsed
Epoch 26,   750/ 1501; acc:   0.00; ppl:  10.64; 2971 src tok/s; 3821 tgt tok/s;  21627 s elapsed
Epoch 26,   800/ 1501; acc:   0.00; ppl:  10.11; 2995 src tok/s; 3900 tgt tok/s;  21657 s elapsed
Epoch 26,   850/ 1501; acc:   0.00; ppl:   9.18; 2950 src tok/s; 3918 tgt tok/s;  21684 s elapsed
Epoch 26,   900/ 1501; acc:   0.00; ppl:  10.35; 3129 src tok/s; 4062 tgt tok/s;  21715 s elapsed
Epoch 26,   950/ 1501; acc:   0.00; ppl:  10.27; 3342 src tok/s; 4298 tgt tok/s;  21743 s elapsed
Epoch 26,  1000/ 1501; acc:   0.00; ppl:  10.61; 3099 src tok/s; 4020 tgt tok/s;  21772 s elapsed
Epoch 26,  1050/ 1501; acc:   0.00; ppl:  10.00; 3151 src tok/s; 4064 tgt tok/s;  21799 s elapsed
Epoch 26,  1100/ 1501; acc:   0.00; ppl:  10.89; 3194 src tok/s; 4053 tgt tok/s;  21829 s elapsed
Epoch 26,  1150/ 1501; acc:   0.00; ppl:  10.16; 3052 src tok/s; 3942 tgt tok/s;  21858 s elapsed
Epoch 26,  1200/ 1501; acc:   0.00; ppl:  10.42; 3087 src tok/s; 3957 tgt tok/s;  21887 s elapsed
Epoch 26,  1250/ 1501; acc:   0.00; ppl:   9.64; 3051 src tok/s; 4026 tgt tok/s;  21913 s elapsed
Epoch 26,  1300/ 1501; acc:   0.00; ppl:  10.31; 3176 src tok/s; 4129 tgt tok/s;  21942 s elapsed
Epoch 26,  1350/ 1501; acc:   0.00; ppl:  10.57; 3199 src tok/s; 4112 tgt tok/s;  21972 s elapsed
Epoch 26,  1400/ 1501; acc:   0.00; ppl:  10.53; 3129 src tok/s; 4007 tgt tok/s;  22004 s elapsed
Epoch 26,  1450/ 1501; acc:   0.00; ppl:  10.36; 3199 src tok/s; 4146 tgt tok/s;  22033 s elapsed
Epoch 26,  1500/ 1501; acc:   0.00; ppl:   9.92; 3257 src tok/s; 4204 tgt tok/s;  22060 s elapsed
Train perplexity: 10.1667
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 1.90735e-06

Epoch 27,    50/ 1501; acc:   0.00; ppl:   9.91; 3237 src tok/s; 4238 tgt tok/s;  22088 s elapsed
Epoch 27,   100/ 1501; acc:   0.00; ppl:   9.88; 3137 src tok/s; 4061 tgt tok/s;  22115 s elapsed
Epoch 27,   150/ 1501; acc:   0.00; ppl:   9.79; 3208 src tok/s; 4149 tgt tok/s;  22140 s elapsed
Epoch 27,   200/ 1501; acc:   0.00; ppl:  10.25; 3243 src tok/s; 4188 tgt tok/s;  22169 s elapsed
Epoch 27,   250/ 1501; acc:   0.00; ppl:  10.70; 3333 src tok/s; 4276 tgt tok/s;  22197 s elapsed
Epoch 27,   300/ 1501; acc:   0.00; ppl:  10.01; 3315 src tok/s; 4286 tgt tok/s;  22225 s elapsed
Epoch 27,   350/ 1501; acc:   0.00; ppl:  10.67; 3427 src tok/s; 4346 tgt tok/s;  22254 s elapsed
Epoch 27,   400/ 1501; acc:   0.00; ppl:  10.38; 3101 src tok/s; 3991 tgt tok/s;  22283 s elapsed
Epoch 27,   450/ 1501; acc:   0.00; ppl:  10.66; 3290 src tok/s; 4202 tgt tok/s;  22313 s elapsed
Epoch 27,   500/ 1501; acc:   0.00; ppl:  10.21; 3275 src tok/s; 4244 tgt tok/s;  22343 s elapsed
Epoch 27,   550/ 1501; acc:   0.00; ppl:   9.86; 3165 src tok/s; 4139 tgt tok/s;  22370 s elapsed
Epoch 27,   600/ 1501; acc:   0.00; ppl:   9.92; 3178 src tok/s; 4152 tgt tok/s;  22397 s elapsed
Epoch 27,   650/ 1501; acc:   0.00; ppl:  10.94; 3534 src tok/s; 4408 tgt tok/s;  22425 s elapsed
Epoch 27,   700/ 1501; acc:   0.00; ppl:  10.33; 3220 src tok/s; 4189 tgt tok/s;  22453 s elapsed
Epoch 27,   750/ 1501; acc:   0.00; ppl:  10.32; 3260 src tok/s; 4197 tgt tok/s;  22481 s elapsed
Epoch 27,   800/ 1501; acc:   0.00; ppl:  10.04; 3193 src tok/s; 4157 tgt tok/s;  22509 s elapsed
Epoch 27,   850/ 1501; acc:   0.00; ppl:  10.65; 3258 src tok/s; 4236 tgt tok/s;  22540 s elapsed
Epoch 27,   900/ 1501; acc:   0.00; ppl:   9.51; 3090 src tok/s; 4073 tgt tok/s;  22565 s elapsed
Epoch 27,   950/ 1501; acc:   0.00; ppl:  10.07; 3213 src tok/s; 4183 tgt tok/s;  22592 s elapsed
Epoch 27,  1000/ 1501; acc:   0.00; ppl:   9.77; 3061 src tok/s; 4032 tgt tok/s;  22620 s elapsed
Epoch 27,  1050/ 1501; acc:   0.00; ppl:  10.67; 3234 src tok/s; 4175 tgt tok/s;  22649 s elapsed
Epoch 27,  1100/ 1501; acc:   0.00; ppl:   9.71; 3217 src tok/s; 4228 tgt tok/s;  22676 s elapsed
Epoch 27,  1150/ 1501; acc:   0.00; ppl:   9.84; 3123 src tok/s; 4066 tgt tok/s;  22702 s elapsed
Epoch 27,  1200/ 1501; acc:   0.00; ppl:  10.29; 3143 src tok/s; 4119 tgt tok/s;  22730 s elapsed
Epoch 27,  1250/ 1501; acc:   0.00; ppl:  10.08; 3174 src tok/s; 4103 tgt tok/s;  22759 s elapsed
Epoch 27,  1300/ 1501; acc:   0.00; ppl:   9.65; 3017 src tok/s; 3944 tgt tok/s;  22786 s elapsed
Epoch 27,  1350/ 1501; acc:   0.00; ppl:   9.58; 3107 src tok/s; 4070 tgt tok/s;  22812 s elapsed
Epoch 27,  1400/ 1501; acc:   0.00; ppl:  10.32; 3152 src tok/s; 4070 tgt tok/s;  22841 s elapsed
Epoch 27,  1450/ 1501; acc:   0.00; ppl:  10.40; 3033 src tok/s; 3933 tgt tok/s;  22870 s elapsed
Epoch 27,  1500/ 1501; acc:   0.00; ppl:  10.41; 3426 src tok/s; 4339 tgt tok/s;  22899 s elapsed
Train perplexity: 10.1713
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 9.53674e-07

Epoch 28,    50/ 1501; acc:   0.00; ppl:  10.46; 3338 src tok/s; 4303 tgt tok/s;  22930 s elapsed
Epoch 28,   100/ 1501; acc:   0.00; ppl:  10.46; 3367 src tok/s; 4318 tgt tok/s;  22957 s elapsed
Epoch 28,   150/ 1501; acc:   0.00; ppl:  10.09; 3324 src tok/s; 4269 tgt tok/s;  22983 s elapsed
Epoch 28,   200/ 1501; acc:   0.00; ppl:   9.67; 3117 src tok/s; 4075 tgt tok/s;  23009 s elapsed
Epoch 28,   250/ 1501; acc:   0.00; ppl:  10.01; 3154 src tok/s; 4146 tgt tok/s;  23036 s elapsed
Epoch 28,   300/ 1501; acc:   0.00; ppl:  10.10; 3290 src tok/s; 4296 tgt tok/s;  23063 s elapsed
Epoch 28,   350/ 1501; acc:   0.00; ppl:   9.75; 3229 src tok/s; 4192 tgt tok/s;  23089 s elapsed
Epoch 28,   400/ 1501; acc:   0.00; ppl:  10.65; 3029 src tok/s; 3900 tgt tok/s;  23122 s elapsed
Epoch 28,   450/ 1501; acc:   0.00; ppl:  10.50; 3053 src tok/s; 3885 tgt tok/s;  23152 s elapsed
Epoch 28,   500/ 1501; acc:   0.00; ppl:   9.60; 2934 src tok/s; 3851 tgt tok/s;  23180 s elapsed
Epoch 28,   550/ 1501; acc:   0.00; ppl:   9.71; 2898 src tok/s; 3791 tgt tok/s;  23210 s elapsed
Epoch 28,   600/ 1501; acc:   0.00; ppl:   9.83; 2903 src tok/s; 3808 tgt tok/s;  23239 s elapsed
Epoch 28,   650/ 1501; acc:   0.00; ppl:  10.41; 3148 src tok/s; 3990 tgt tok/s;  23270 s elapsed
Epoch 28,   700/ 1501; acc:   0.00; ppl:  10.41; 3325 src tok/s; 4279 tgt tok/s;  23298 s elapsed
Epoch 28,   750/ 1501; acc:   0.00; ppl:  10.44; 3277 src tok/s; 4234 tgt tok/s;  23327 s elapsed
Epoch 28,   800/ 1501; acc:   0.00; ppl:  10.88; 3257 src tok/s; 4193 tgt tok/s;  23356 s elapsed
Epoch 28,   850/ 1501; acc:   0.00; ppl:  10.71; 3343 src tok/s; 4274 tgt tok/s;  23387 s elapsed
Epoch 28,   900/ 1501; acc:   0.00; ppl:  10.37; 3269 src tok/s; 4184 tgt tok/s;  23416 s elapsed
Epoch 28,   950/ 1501; acc:   0.00; ppl:  10.29; 3144 src tok/s; 4026 tgt tok/s;  23444 s elapsed
Epoch 28,  1000/ 1501; acc:   0.00; ppl:   9.85; 3146 src tok/s; 4097 tgt tok/s;  23472 s elapsed
Epoch 28,  1050/ 1501; acc:   0.00; ppl:  10.00; 3216 src tok/s; 4179 tgt tok/s;  23499 s elapsed
Epoch 28,  1100/ 1501; acc:   0.00; ppl:  10.32; 3073 src tok/s; 4037 tgt tok/s;  23527 s elapsed
Epoch 28,  1150/ 1501; acc:   0.00; ppl:   9.89; 3038 src tok/s; 3935 tgt tok/s;  23557 s elapsed
Epoch 28,  1200/ 1501; acc:   0.00; ppl:  10.12; 3060 src tok/s; 3982 tgt tok/s;  23587 s elapsed
Epoch 28,  1250/ 1501; acc:   0.00; ppl:  10.06; 3042 src tok/s; 3942 tgt tok/s;  23616 s elapsed
Epoch 28,  1300/ 1501; acc:   0.00; ppl:  10.53; 3232 src tok/s; 4165 tgt tok/s;  23644 s elapsed
Epoch 28,  1350/ 1501; acc:   0.00; ppl:   9.82; 3183 src tok/s; 4132 tgt tok/s;  23671 s elapsed
Epoch 28,  1400/ 1501; acc:   0.00; ppl:  10.28; 3128 src tok/s; 4083 tgt tok/s;  23699 s elapsed
Epoch 28,  1450/ 1501; acc:   0.00; ppl:   9.74; 3033 src tok/s; 3978 tgt tok/s;  23726 s elapsed
Epoch 28,  1500/ 1501; acc:   0.00; ppl:  10.28; 3303 src tok/s; 4281 tgt tok/s;  23753 s elapsed
Train perplexity: 10.1821
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 4.76837e-07

Epoch 29,    50/ 1501; acc:   0.00; ppl:  10.21; 3209 src tok/s; 4173 tgt tok/s;  23783 s elapsed
Epoch 29,   100/ 1501; acc:   0.00; ppl:   9.17; 3109 src tok/s; 4113 tgt tok/s;  23808 s elapsed
Epoch 29,   150/ 1501; acc:   0.00; ppl:  10.28; 3233 src tok/s; 4167 tgt tok/s;  23836 s elapsed
Epoch 29,   200/ 1501; acc:   0.00; ppl:  10.25; 3343 src tok/s; 4331 tgt tok/s;  23863 s elapsed
Epoch 29,   250/ 1501; acc:   0.00; ppl:  10.40; 3272 src tok/s; 4196 tgt tok/s;  23892 s elapsed
Epoch 29,   300/ 1501; acc:   0.00; ppl:  10.67; 3363 src tok/s; 4305 tgt tok/s;  23919 s elapsed
Epoch 29,   350/ 1501; acc:   0.00; ppl:   9.69; 3195 src tok/s; 4180 tgt tok/s;  23945 s elapsed
Epoch 29,   400/ 1501; acc:   0.00; ppl:  10.57; 3417 src tok/s; 4334 tgt tok/s;  23974 s elapsed
Epoch 29,   450/ 1501; acc:   0.00; ppl:  10.57; 3314 src tok/s; 4265 tgt tok/s;  24004 s elapsed
Epoch 29,   500/ 1501; acc:   0.00; ppl:  10.34; 3253 src tok/s; 4170 tgt tok/s;  24032 s elapsed
Epoch 29,   550/ 1501; acc:   0.00; ppl:   9.04; 3130 src tok/s; 4135 tgt tok/s;  24056 s elapsed
Epoch 29,   600/ 1501; acc:   0.00; ppl:  10.57; 3287 src tok/s; 4212 tgt tok/s;  24085 s elapsed
Epoch 29,   650/ 1501; acc:   0.00; ppl:  10.78; 3170 src tok/s; 4068 tgt tok/s;  24116 s elapsed
Epoch 29,   700/ 1501; acc:   0.00; ppl:   9.62; 2021 src tok/s; 2664 tgt tok/s;  24158 s elapsed
Epoch 29,   750/ 1501; acc:   0.00; ppl:   9.95; 2326 src tok/s; 3051 tgt tok/s;  24195 s elapsed
Epoch 29,   800/ 1501; acc:   0.00; ppl:  10.48; 3289 src tok/s; 4219 tgt tok/s;  24224 s elapsed
Epoch 29,   850/ 1501; acc:   0.00; ppl:  10.81; 3332 src tok/s; 4216 tgt tok/s;  24254 s elapsed
Epoch 29,   900/ 1501; acc:   0.00; ppl:   9.91; 3123 src tok/s; 4048 tgt tok/s;  24281 s elapsed
Epoch 29,   950/ 1501; acc:   0.00; ppl:  10.66; 3286 src tok/s; 4272 tgt tok/s;  24310 s elapsed
Epoch 29,  1000/ 1501; acc:   0.00; ppl:   9.42; 3262 src tok/s; 4275 tgt tok/s;  24335 s elapsed
Epoch 29,  1050/ 1501; acc:   0.00; ppl:   9.54; 3058 src tok/s; 4026 tgt tok/s;  24361 s elapsed
Epoch 29,  1100/ 1501; acc:   0.00; ppl:   9.21; 3146 src tok/s; 4145 tgt tok/s;  24386 s elapsed
Epoch 29,  1150/ 1501; acc:   0.00; ppl:  10.85; 3362 src tok/s; 4309 tgt tok/s;  24415 s elapsed
Epoch 29,  1200/ 1501; acc:   0.00; ppl:  10.59; 3345 src tok/s; 4282 tgt tok/s;  24445 s elapsed
Epoch 29,  1250/ 1501; acc:   0.00; ppl:  10.51; 3160 src tok/s; 4114 tgt tok/s;  24473 s elapsed
Epoch 29,  1300/ 1501; acc:   0.00; ppl:  10.17; 3278 src tok/s; 4239 tgt tok/s;  24500 s elapsed
Epoch 29,  1350/ 1501; acc:   0.00; ppl:  10.09; 3203 src tok/s; 4166 tgt tok/s;  24528 s elapsed
Epoch 29,  1400/ 1501; acc:   0.00; ppl:  10.40; 3283 src tok/s; 4254 tgt tok/s;  24557 s elapsed
Epoch 29,  1450/ 1501; acc:   0.00; ppl:  10.16; 3187 src tok/s; 4143 tgt tok/s;  24585 s elapsed
Epoch 29,  1500/ 1501; acc:   0.00; ppl:   9.56; 3186 src tok/s; 4132 tgt tok/s;  24611 s elapsed
Train perplexity: 10.1709
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 2.38419e-07

Epoch 30,    50/ 1501; acc:   0.00; ppl:   9.62; 3178 src tok/s; 4162 tgt tok/s;  24641 s elapsed
Epoch 30,   100/ 1501; acc:   0.00; ppl:   9.77; 3274 src tok/s; 4251 tgt tok/s;  24668 s elapsed
Epoch 30,   150/ 1501; acc:   0.00; ppl:  10.76; 3219 src tok/s; 4135 tgt tok/s;  24697 s elapsed
Epoch 30,   200/ 1501; acc:   0.00; ppl:   9.84; 2974 src tok/s; 3922 tgt tok/s;  24724 s elapsed
Epoch 30,   250/ 1501; acc:   0.00; ppl:  10.08; 3147 src tok/s; 4086 tgt tok/s;  24752 s elapsed
Epoch 30,   300/ 1501; acc:   0.00; ppl:  10.75; 3370 src tok/s; 4297 tgt tok/s;  24782 s elapsed
Epoch 30,   350/ 1501; acc:   0.00; ppl:  10.28; 3143 src tok/s; 4097 tgt tok/s;  24811 s elapsed
Epoch 30,   400/ 1501; acc:   0.00; ppl:  10.38; 3256 src tok/s; 4185 tgt tok/s;  24838 s elapsed
Epoch 30,   450/ 1501; acc:   0.00; ppl:  10.62; 3320 src tok/s; 4236 tgt tok/s;  24869 s elapsed
Epoch 30,   500/ 1501; acc:   0.00; ppl:   8.95; 3021 src tok/s; 4015 tgt tok/s;  24894 s elapsed
Epoch 30,   550/ 1501; acc:   0.00; ppl:   9.85; 3149 src tok/s; 4132 tgt tok/s;  24922 s elapsed
Epoch 30,   600/ 1501; acc:   0.00; ppl:  10.09; 3240 src tok/s; 4217 tgt tok/s;  24951 s elapsed
Epoch 30,   650/ 1501; acc:   0.00; ppl:  10.47; 3241 src tok/s; 4135 tgt tok/s;  24977 s elapsed
Epoch 30,   700/ 1501; acc:   0.00; ppl:  10.95; 3244 src tok/s; 4143 tgt tok/s;  25006 s elapsed
Epoch 30,   750/ 1501; acc:   0.00; ppl:  10.31; 3061 src tok/s; 3978 tgt tok/s;  25037 s elapsed
Epoch 30,   800/ 1501; acc:   0.00; ppl:  10.26; 3279 src tok/s; 4268 tgt tok/s;  25063 s elapsed
Epoch 30,   850/ 1501; acc:   0.00; ppl:   9.95; 3142 src tok/s; 4106 tgt tok/s;  25090 s elapsed
Epoch 30,   900/ 1501; acc:   0.00; ppl:  10.33; 3312 src tok/s; 4269 tgt tok/s;  25118 s elapsed
Epoch 30,   950/ 1501; acc:   0.00; ppl:  10.44; 3265 src tok/s; 4235 tgt tok/s;  25146 s elapsed
Epoch 30,  1000/ 1501; acc:   0.00; ppl:   9.64; 3176 src tok/s; 4163 tgt tok/s;  25173 s elapsed
Epoch 30,  1050/ 1501; acc:   0.00; ppl:  10.34; 3262 src tok/s; 4199 tgt tok/s;  25200 s elapsed
Epoch 30,  1100/ 1501; acc:   0.00; ppl:  10.09; 3315 src tok/s; 4251 tgt tok/s;  25227 s elapsed
Epoch 30,  1150/ 1501; acc:   0.00; ppl:  10.36; 3131 src tok/s; 4039 tgt tok/s;  25256 s elapsed
Epoch 30,  1200/ 1501; acc:   0.00; ppl:   9.53; 2996 src tok/s; 3948 tgt tok/s;  25283 s elapsed
Epoch 30,  1250/ 1501; acc:   0.00; ppl:   9.92; 3190 src tok/s; 4146 tgt tok/s;  25311 s elapsed
Epoch 30,  1300/ 1501; acc:   0.00; ppl:  10.11; 3219 src tok/s; 4187 tgt tok/s;  25338 s elapsed
Epoch 30,  1350/ 1501; acc:   0.00; ppl:  10.68; 3367 src tok/s; 4270 tgt tok/s;  25367 s elapsed
Epoch 30,  1400/ 1501; acc:   0.00; ppl:   9.70; 3238 src tok/s; 4212 tgt tok/s;  25393 s elapsed
Epoch 30,  1450/ 1501; acc:   0.00; ppl:  10.25; 3237 src tok/s; 4170 tgt tok/s;  25422 s elapsed
Epoch 30,  1500/ 1501; acc:   0.00; ppl:  10.67; 3345 src tok/s; 4281 tgt tok/s;  25451 s elapsed
Train perplexity: 10.1753
Train accuracy: 0
Validation perplexity: 19.0265
Validation accuracy: 0
Decaying learning rate to 1.19209e-07
