// Copyright Teleportd Ltd. and other Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "nn.hh"

#include <sstream>
#include <algorithm>
#include <node_object_wrap.h>
#include <node.h>

using namespace v8;
using namespace node;
using namespace std;

Persistent<Function> NN::constructor;

/******************************************************************************/
/*                             NN IMPLEMENTATION                              */
/******************************************************************************/

//
// ### NN
// ```
// @layers {vector<int>} the layers structure
// @alpha  {double} the learning rate
// @beta   {double} the momentum
// ```
//
NN::NN(vector<int> &layers,
       double alpha = 0.3,
       double beta = 0.1,
       double bias = -1.0)
{
  log_ = false;
  layers_ = layers;
  alpha_ = alpha;
  beta_ = beta;
  bias_ = bias;
  op_count_ = 0;
  L_ = layers.size();

  /* Layers initialization */
  W_.resize(L_); dW_.resize(L_); B_.resize(L_);
  D_.resize(L_); sum_.resize(L_); val_.resize(L_);

  for(int l = 0; l < L_; l++) {
    W_[l].resize(layers_[l]);
    dW_[l].resize(layers_[l]);
    B_[l].resize(layers_[l]);

    D_[l].resize(layers_[l]);
    sum_[l].resize(layers_[l]);
    val_[l].resize(layers_[l]);

    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        W_[l][i].resize(layers_[l-1]);
        dW_[l][i].resize(layers_[l-1]);
        B_[l][i] = NN::fRand(0.2, 0.4);

        sum_[l][i] = 0;
        val_[l][i] = 0;

        for(int j = 0; j < layers_[l-1]; j++) {
          W_[l][i][j] = NN::fRand(0.2, 0.4);
          dW_[l][i][j] = 0;
        }
      }
    }
  }
}

//
// ### NN
// ```
// @str {std::string} the string representation of the network
// ```
//
NN::NN(std::string& str)
{
  log_ = false;
  istringstream iss(str);

  iss >> L_;
  layers_.resize(L_);

  for(int i = 0; i < (int)layers_.size(); i ++) {
    iss >> layers_[i];
  }

  iss >> alpha_;
  iss >> beta_;
  iss >> bias_;
  op_count_ = 0;

  /* Layers initialization */
  W_.resize(L_); dW_.resize(L_); B_.resize(L_);
  D_.resize(L_); sum_.resize(L_); val_.resize(L_);

  for(int l = 0; l < L_; l++) {
    W_[l].resize(layers_[l]);
    dW_[l].resize(layers_[l]);
    B_[l].resize(layers_[l]);

    D_[l].resize(layers_[l]);
    sum_[l].resize(layers_[l]);
    val_[l].resize(layers_[l]);

    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        W_[l][i].resize(layers_[l-1]);
        dW_[l][i].resize(layers_[l-1]);
        iss >> B_[l][i];

        sum_[l][i] = 0;
        val_[l][i] = 0;

        for(int j = 0; j < layers_[l-1]; j++) {
          iss >> W_[l][i][j];
          dW_[l][i][j] = 0;
        }
      }
    }
  }
}

//
// ### NN Copy constructor
// ```
// @nn <NN> the NN to copy
// ```
//
NN::NN(NN const& nn)
{
  layers_ = vector<int>(nn.layers_);
  alpha_ = double(nn.alpha_);
  beta_ = double(nn.beta_);
  bias_ = double(nn.bias_);
  op_count_ = long(nn.op_count_);
  L_ = int(layers_.size());

  W_.resize(L_); dW_.resize(L_); B_.resize(L_);
  D_.resize(L_); sum_.resize(L_); val_.resize(L_);

  for(int l = 0; l < L_; l++) {
    W_[l].resize(layers_[l]);
    dW_[l].resize(layers_[l]);
    B_[l].resize(layers_[l]);

    D_[l].resize(layers_[l]);
    sum_[l].resize(layers_[l]);
    val_[l].resize(layers_[l]);

    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        W_[l][i].resize(layers_[l-1]);
        dW_[l][i].resize(layers_[l-1]);

        /* Initialize values */
        B_[l][i] = nn.B_[l][i];
        D_[l][i] = nn.D_[l][i];
        sum_[l][i] = nn.sum_[l][i];
        val_[l][i] = nn.val_[l][i];

        for(int j = 0; j < layers_[l-1]; j++) {
          W_[l][i][j] = nn.W_[l][i][j];
          dW_[l][i][j] = nn.dW_[l][i][j];
        }
      }
    }
  }
}

//
// ### ~NN
//
NN::~NN() {};

//
// ### fRand
// ```
// @min {double} lower bound
// @max {double} upper bound
// ```
//
double NN::fRand(double fMin, double fMax)
{
  double f = (double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
};


//
// ### run
// ```
// @in {vector<double>} input vector
// ```
//
vector<double> NN::run(vector<double> &in)
{
  /* initialization */
  if(in.size() != (unsigned)layers_[0]) {
    cout << "Incompatible Dimensions `in` (" << in.size() << ")" << endl;
  }
  for(unsigned int i = 0; i < val_[0].size(); i++) {
    val_[0][i] = in[i];
  }

  /* propagation */
  for(int l = 1; l < L_; l++) {
    for(int i = 0; i < layers_[l]; i++) {
      sum_[l][i] = bias_ * B_[l][i];
      for(int j = 0; j < layers_[l-1]; j++) {
        sum_[l][i] += W_[l][i][j] * val_[l-1][j];
      }
      val_[l][i] = 1 / (1 + exp(-sum_[l][i]));
    }
  }

  return val_[L_-1];
}


//
// ### learn
// ```
// @in {vector<double>} input vector
// @out {vector<double>} result vector
// ```
//
vector<double> NN::learn(vector<double> &in,
                         vector<double> &out)
{
  if(out.size() != (unsigned)layers_[L_-1]) {
    cout << "Incompatible Dimensions `out` (" << out.size() << ")" << endl;
  }

  this->run(in);

  /* back propagation */
  for(int l = L_-1; l >= 0; l--) {
    for(int j = 0; j < layers_[l]; j++) {
      /* output layer */
      if(l == L_-1) {
        D_[l][j] = out[j] - val_[l][j];
      }
      /* inner layer */
      else {
        D_[l][j] = 0;
        for(int i = 0; i < layers_[l+1]; i++) {
          if(l > 0) {
            D_[l][j] += W_[l+1][i][j] * D_[l+1][i];
          }
          /* weight update */
          double dW = alpha_ * val_[l][j] * D_[l+1][i];

          W_[l+1][i][j] += dW + beta_ * dW_[l+1][i][j];
          dW_[l+1][i][j] = dW;

          /* bias weight update */
          B_[l+1][i] = alpha_ * bias_ * D_[l+1][i];

          op_count_++;
        }
      }
      if(l > 0) {
        D_[l][j] *= val_[l][j] * (1 - val_[l][j]);
      }
    }
  }

  return val_[L_-1];
}

//
// ### learn_step
// Learn the current training set and return mean square error
//
double NN::learn_step() {
  double err = 0.0;

  for(unsigned int i = 0; i < train_in_.size(); i++) {
    vector<double> res = this->learn(train_in_[i], train_out_[i]);
    /* error calculation */
    double e = 0;
    for(unsigned int j = 0; j < res.size(); j++) {
      e += pow(res[j] - train_out_[i][j], 2);
    }
    err += e / res.size();
  }

  return err;
}


//
// ### train_set_add
// ```
// @in         {vector<double>} input vector
// @out        {vector<double>} result vector to learn on
// ```
//
void NN::train_set_add(vector<double> &in,
                       vector<double> &out)
{
  /* initialization */
  if(in.size() != (unsigned)layers_[0]) {
    cout << "Incompatible Dimensions `in` (" << in.size() << ")" << endl;
  }
  if(out.size() != (unsigned)layers_[L_-1]) {
    cout << "Incompatible Dimensions `out` (" << out.size() << ")" << endl;
  }

  train_in_.push_back(in);
  train_out_.push_back(out);
}

//
// ### train_set_clear
//
void NN::train_set_clear()
{
  train_in_.clear();
  train_out_.clear();
}

//
// ### train
// ```
// @error      {double} target error
// @iterations {int} max number of iterations
// ```
//
void NN::train(double error = 0.01,
               int iterations = 20000)
{
  if(train_out_.size() != train_in_.size()) {
    cout << "Incompatible Dimensions `train_out_` ("
         << train_out_.size() << ")"
         << " `train_in_` (" << train_in_.size() << ")" << endl;
  }

  if(log_) {
    cout << "----------------------------------" << endl;
    cout << "  LAYERS: [";
    for(unsigned int i = 0; i < layers_.size(); i++) {
      if(i > 0) cout << ", ";
      cout << layers_[i];
    }
    cout << "]" << endl;
    cout << "  ALPHA: " << alpha_ << endl;
    cout << "  BETA: " << beta_ << endl;
    cout << "  BIAS: " << bias_ << endl;
    cout << "  TRAINING SIZE: " << train_in_.size() << endl;
    cout << "  ERROR THRESHOLD: " << error << endl;
    cout << "  MAX ITERATIONS: " << iterations << endl;
    cout << "----------------------------------" << endl;
  }
  int it = 0;
  double err = 0;

  do {
    err = 0;
    for(unsigned int i = 0; i < train_in_.size(); i++) {
      vector<double> res = this->learn(train_in_[i], train_out_[i]);
      /* error calculation */
      double e = 0;
      for(unsigned int j = 0; j < res.size(); j++) {
        e += pow(res[j] - train_out_[i][j], 2);
      }
      err += e / res.size();
    }
    err /= train_in_.size();
    it++;
    if(log_) {
      cout << "[" << it << "] " << err << endl;
    }
  } while(err > error && it < iterations);
}

//
// ### mt_train
// ```
// @error      {double} target error
// @iterations {int} max number of iterations
// @step_size  {int} size of training set by step
// @n_threads  {int} the number of threads to use
// ```
//
void NN::mt_train(double error = 0.01,
                  int iterations = 20000,
                  int step_size = 100,
                  int thread = 4)
{
  if(train_out_.size() != train_in_.size()) {
    cout << "Incompatible Dimensions `train_out_` ("
         << train_out_.size() << ")"
         << " `train_in_` (" << train_in_.size() << ")" << endl;
  }

  int it = 0;
  double err = 0.0;

  int step = 0;
  int total = 0;

  if(log_) {
    cout << "----------------------------------" << endl;
    cout << "  STARTING MULTITHREAD TRAINING" << endl << endl;
  }
  if(train_in_.size() < 1) {
    cout << "Training set is empty..." << endl;
  }
  else {
    if(log_) {
      cout << "  STEP SIZE: " << step_size << endl;
      cout << "  NUMBER OF THREADS: " << thread << endl << endl;
      cout << "  ERROR THRESHOLD: " << error << endl;
      cout << "  MAX ITERATIONS: " << iterations << endl << endl;
      cout << "  ALPHA: " << alpha_ << endl;
      cout << "  BETA: " << beta_ << endl;
      cout << "  BIAS: " << bias_ << endl;
      cout << "  TRAINING SIZE: " << train_in_.size() << endl;
      cout << "----------------------------------" << endl;
    }

    /* Main iteration loop */
    do {
      step = 0;
      total = 0;
      err = 0.0;

      /* Step loop */
      while(total < (int)train_in_.size()) {
        //cout << endl << "IT[" << it << "] Step " << step << endl;
        //cout << "Initializing children NNs... ";
        /* Copy and initialize children NN */
        NN** nns = new NN*[thread];
        for(int i = 0; i < thread; i++) {
          nns[i] = new NN(*this);
          nns[i]->train_set_clear();
        }
        //cout << "OK!" << endl <<
        //  "Split data... ";
        int added = MT_NN::split_data(nns, thread, step, step_size,
                                      train_in_,
                                      train_out_);
        total += added;
        //cout << "OK!" << endl;

        int n_thread = thread;
        /* If we splited less training points than number of NN, we only work */
        /* with this number of NN during the current step                     */
        if(added < n_thread) {
          n_thread = added;
        }

        /* Launch threads */
        //cout << "Launch threads... ";
        uv_thread_t* nns_ids = new uv_thread_t[n_thread];
        MT_NN::LearnWorker** workers = new MT_NN::LearnWorker*[n_thread];
        for(int i = 0; i < n_thread; i++) {
          workers[i] = new MT_NN::LearnWorker();
          workers[i]->nn = nns[i];

          uv_thread_create(&nns_ids[i], MT_NN::learn, workers[i]);
        }
        //cout << "OK!" << endl;

        /* Wait until threads are done */
        //cout << "Waiting they are done... ";
        for(int i = 0; i < n_thread; i++) {
          uv_thread_join(&nns_ids[i]);
        }
        //cout << "OK!" << endl;

        /* Compute result */
        //cout << "Compute results... ";
        NN *nn_origin = new NN(*this);
        for(int i = 0; i < n_thread; i++) {
          *this += *nns[i];
        }
        *this -= *nn_origin;
        *this /= n_thread;
        //cout << "OK!" << endl;

        /* look at error & it */
        //cout << "Compute error... ";
        int total_training_size = 0;
        for(int i = 0; i < n_thread; i++) {
          err += workers[i]->error;
          total_training_size += workers[i]->nn->train_in_.size();
        }
        err /= total_training_size;
        //cout << "OK!" << endl;

        /* Free memory */
        //cout << "Free memory... ";
        delete nn_origin;
        for(int i = 0; i < n_thread; i++) {
          delete nns[i];
          delete workers[i];
        }
        //cout << "OK!" << endl;

        step++;
      }
      if(log_) {
        cout << "[" << it << "] " << err << endl;
      }
      it++;
    } while(err > error && it < iterations);
  }
}

//
// ### to_string
//
std::string NN::to_string()
{
  ostringstream oss;

  oss << (int)layers_.size();

  for(int i = 0; i < (int)layers_.size(); i ++) {
    oss << " " << layers_[i];
  }

  oss << " " << alpha_;
  oss << " " << beta_;
  oss << " " << bias_;

  for(int l = 0; l < L_; l++) {
    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        oss << " " << B_[l][i];
        for(int j = 0; j < layers_[l-1]; j++) {
          oss << " " << W_[l][i][j];
        }
      }
    }
  }

  return oss.str();
}

//
// ### get_state
//
std::string NN::get_state(bool compact = false)
{
  ostringstream oss;

  oss << (int)layers_.size();

  for(int i = 0; i < (int)layers_.size(); i++) {
    oss << " " << layers_[i];
  }

  if(compact)
    oss << " " << "compact";
  else
    oss << " " << "full";

  for(int l = 0; l < L_; l++) {
    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        for(int j = 0; j < layers_[l-1]; j++) {
          double s = W_[l][i][j] * val_[l-1][j];
          if(!compact) {
            oss << " " << s;
          }
          else if(s != 0.0) {
            oss << " " << l
                << " " << i
                << " " << j
                << " " << s;
          }
        }
      }
    }
  }

  return oss.str();
}

//
// ### set_log
//
void NN::set_log(bool status)
{
  log_ = status;
}

/******************************************************************************/
/*                                 OPERATORS                                  */
/******************************************************************************/

//
// ### operator+=
//
NN& NN::operator+=(NN const& nn) {
  if(L_ != nn.L_) {
    cout << "Can't add different layers " << L_ << " & " << nn.L_ << endl;
    return *this;
  }

  /* Add Weights */
  for(int l = 0; l < L_; l++) {
    if(layers_[l] != nn.layers_[l]) {
      cout << "Can't add different layers" << endl;
      return *this;
    }

    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        B_[l][i] += nn.B_[l][i];

        for(int j = 0; j < layers_[l-1]; j++) {
          W_[l][i][j] += nn.W_[l][i][j];
        }
      }
    }
  }

  return *this;
}

//
// ### operator-=
//
NN& NN::operator-=(NN const& nn) {
  if(L_ != nn.L_) {
    cout << "Can't substract different layers " << L_ << " & " << nn.L_ << endl;
    return *this;
  }

  /* Add Weights */
  for(int l = 0; l < L_; l++) {
    if(layers_[l] != nn.layers_[l]) {
      cout << "Can't substract different layers" << endl;
      return *this;
    }

    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        B_[l][i] -= nn.B_[l][i];

        for(int j = 0; j < layers_[l-1]; j++) {
          W_[l][i][j] -= nn.W_[l][i][j];
        }
      }
    }
  }

  return *this;
}

//
// ### operator/=
//
NN& NN::operator/=(int const& N) {
  /* Add Weights */
  for(int l = 0; l < L_; l++) {
    for(int i = 0; i < layers_[l]; i++) {
      if(l > 0) {
        B_[l][i] /= N;

        for(int j = 0; j < layers_[l-1]; j++) {
          W_[l][i][j] /= N;
        }
      }
    }
  }

  return *this;
}

/******************************************************************************/
/*                             NN BINDING                                     */
/******************************************************************************/

//
// ### ToString wrapper
//
void NN::ToString(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  /* return values */
  Local<String> result = String::NewFromUtf8(isolate, nn->to_string().c_str());

  args.GetReturnValue().Set(result);
}

//
// ### GetState wrapper
//
void NN::GetState(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  bool compact = false;
  if(args[0]->IsBoolean()) {
    compact = args[0]->ToBoolean()->Value();
  }

  /* return values */
  Local<String> result = String::NewFromUtf8(isolate, nn->get_state(compact).c_str());

  args.GetReturnValue().Set(result);
}

//
// ### TrainSetAdd wrapper
//
void NN::TrainSetAdd(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  if(!args[0]->IsArray()) {
    isolate->ThrowException(
      Exception::TypeError(
        String::NewFromUtf8(isolate, "Training `in` values expected as argument 0")));

    args.GetReturnValue().SetUndefined();

    return;
  }
  if(!args[1]->IsArray()) {
    isolate->ThrowException(
      Exception::TypeError(
        String::NewFromUtf8(isolate, "Training `out` values expected as argument 0")));

    args.GetReturnValue().SetUndefined();

    return;
  }

  /* training set extraction */
  Local<Array> in = Local<Array>::Cast(args[0]);
  Local<Array> out = Local<Array>::Cast(args[1]);

  vector<double> input(in->Length());
  vector<double> output(out->Length());

  for(unsigned int i = 0; i < in->Length(); i ++) {
    input[i] = in->Get(Integer::New(isolate, i))->ToNumber()->Value();
  }
  for(unsigned int i = 0; i < out->Length(); i ++) {
    output[i] = out->Get(Integer::New(isolate, i))->ToNumber()->Value();
  }

  nn->train_set_add(input, output);

  args.GetReturnValue().SetUndefined();
}

//
// ### Train wrapper
//
void NN::Train(const FunctionCallbackInfo<v8::Value>& args) {
  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  /* call */
  if(args[0]->IsNumber() && args[1]->IsNumber()) {
    nn->train(args[0]->ToNumber()->Value(),
              (int)args[1]->ToNumber()->Value());
  }
  else if(args[0]->IsNumber()) {
    nn->train(args[0]->NumberValue());
  }
  else {
    nn->train();
  }

  args.GetReturnValue().SetUndefined();
}

//
// ### Multithread Train Wrapper
//
void NN::MTTrain(const FunctionCallbackInfo<v8::Value>& args) {
  double target_error = 0;
  int iterations = 0;
  int step_size = 0;
  int threads = 0;

  Local<Function> cb;

  if(args[0]->IsNumber() && args[1]->IsNumber() &&
     args[2]->IsNumber() && args[3]->IsNumber()) {
    target_error = args[0]->NumberValue();
    iterations = (int)args[1]->NumberValue();
    step_size = (int)args[2]->NumberValue();
    threads = (int)args[3]->NumberValue();

    cb = Local<Function>::Cast(args[4]);
  }
  else if(args[0]->IsNumber() && args[1]->IsNumber() && args[2]->IsNumber()) {
    target_error = args[0]->NumberValue();
    iterations = (int)args[1]->NumberValue();
    step_size = (int)args[2]->NumberValue();

    cb = Local<Function>::Cast(args[3]);
  }
  else if(args[0]->IsNumber() && args[1]->IsNumber()) {
    target_error = args[0]->NumberValue();
    iterations = (int)args[1]->NumberValue();

    cb = Local<Function>::Cast(args[2]);
  }
  else if(args[0]->IsNumber()) {
    target_error = args[0]->NumberValue();

    cb = Local<Function>::Cast(args[1]);
  }
  else {
    cb = Local<Function>::Cast(args[0]);
  }

  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  MT_NN::TrainWorker *worker = new MT_NN::TrainWorker();

  worker->request.data = worker;
  worker->cb = cb;
  worker->nn = nn;

  worker->target_error = target_error;
  worker->iterations = iterations;
  worker->step_size = step_size;
  worker->threads = threads;

  uv_queue_work(uv_default_loop(), &worker->request,
                MT_NN::train_start, MT_NN::train_done);

  args.GetReturnValue().SetUndefined();
}


//
// ### Run wrapper
//
void NN::Run(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  /* unwrapping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  if(!args[0]->IsArray()) {
    isolate->ThrowException(
      Exception::TypeError(String::NewFromUtf8(isolate, "Input expected as argument 0")));
    args.GetReturnValue().SetUndefined();

    return;
  }

  Local<Array> l = Local<Array>::Cast(args[0]);
  vector<double> input(l->Length());

  for(unsigned int i = 0; i < l->Length(); i ++) {
    input[i] = l->Get(Integer::New(isolate, i))->NumberValue();
  }

  /* call */
  vector<double> out = nn->run(input);

  /* return values */
  Local<Array> result = Array::New(isolate, out.size());

  for (size_t i = 0; i < out.size(); i++)
    result->Set(Integer::New(isolate, i), Number::New(isolate, out[i]));

  args.GetReturnValue().Set(result);
}

//
// ### New
//
void NN::New(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  NN* nn = NULL;

  if(args[0]->IsString()) {
    std::string str = std::string(
        *v8::String::Utf8Value(args[0]->ToString()));

    nn = new NN(str);
  }

  else if(args[0]->IsArray()) {
    Local<Array> l = Local<Array>::Cast(args[0]);

    if(l->Length() < 2) {
      isolate->ThrowException(
        Exception::TypeError(String::NewFromUtf8(isolate, "Less than 2 Layers")));
      args.GetReturnValue().SetUndefined();

      return;
    }

    vector<int> layers(l->Length());
    for(unsigned int i = 0; i < l->Length(); i++) {
      layers[i] = l->Get(Integer::New(isolate, i))->Int32Value();
    }

    nn = new NN(layers);
  }

  else {
    isolate->ThrowException(
      Exception::TypeError(String::NewFromUtf8(isolate, "Layers expected as argument 0")));

    args.GetReturnValue().SetUndefined();

    return;
  }

  /* wrapping */
  nn->Wrap(args.This());

  args.GetReturnValue().Set(args.This());
}

//
// ### SetLog
//
void NN::SetLog(const FunctionCallbackInfo<v8::Value>& args) {
  Isolate* isolate = args.GetIsolate();

  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  bool status = false;
  if(args[0]->IsBoolean()) {
    status = args[0]->ToBoolean()->Value();
  }
  else {
    isolate->ThrowException(
      Exception::TypeError(String::NewFromUtf8(isolate, "Boolean expected as argument 0")));
    args.GetReturnValue().SetUndefined();

    return;
  }

  nn->set_log(status);

  args.GetReturnValue().SetUndefined();
}

/******************************************************************************/
/*                            MODULE INIT                                     */
/******************************************************************************/

//
// ### Init
//
void NN::Init(Local<Object> exports)
{
  Isolate* isolate = exports->GetIsolate();

  Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
  tpl->SetClassName(String::NewFromUtf8(isolate, "NN"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  NODE_SET_PROTOTYPE_METHOD(tpl, "train_set_add", TrainSetAdd);
  NODE_SET_PROTOTYPE_METHOD(tpl, "train", Train);
  NODE_SET_PROTOTYPE_METHOD(tpl, "mt_train", MTTrain);
  NODE_SET_PROTOTYPE_METHOD(tpl, "run", Run);
  NODE_SET_PROTOTYPE_METHOD(tpl, "to_string", ToString);
  NODE_SET_PROTOTYPE_METHOD(tpl, "get_state", GetState);
  NODE_SET_PROTOTYPE_METHOD(tpl, "set_log", SetLog);

  constructor.Reset(isolate, tpl->GetFunction());

  exports->Set(String::NewFromUtf8(isolate, "NN"), tpl->GetFunction());
}

void InitAll(Local<Object> exports) {
  NN::Init(exports);
}

NODE_MODULE(nn, InitAll)


/******************************************************************************/
/*                           MULTITHREAD TRAINING                             */
/******************************************************************************/

//
// ### train_start
// Start the multithreaded training session. Takes care of creation threads and
// split data among the NNs created for each thread. Once these NN have run an
// learning iteration, it computes the global changes and start again.
//
void MT_NN::train_start(uv_work_t* req) {
  TrainWorker* worker = static_cast<TrainWorker*>(req->data);

  NN* nn = worker->nn;

  /* Parameters */
  double target_error = worker->target_error;
  int iterations = worker->iterations;
  int step_size = worker->step_size;
  int threads = worker->threads;

  if(target_error > 0 && iterations > 0 && step_size > 0 && threads > 0) {
    nn->mt_train(target_error, iterations, step_size, threads);
  }
  else if (target_error > 0 && iterations > 0 && step_size > 0) {
    nn->mt_train(target_error, iterations, step_size);
  }
  else if(target_error > 0 && iterations > 0) {
    nn->mt_train(target_error, iterations);
  }
  else if(target_error > 0) {
    nn->mt_train(target_error);
  }
  else {
    nn->mt_train();
  }
}

//
// ### train_done
//
void MT_NN::train_done(uv_work_t* req, int status) {
  Isolate* isolate = Isolate::GetCurrent();

  TrainWorker* worker = static_cast<TrainWorker*>(req->data);

  if(!worker->error_message.empty()) {
    Local<Value> err = Exception::Error(
                         String::NewFromUtf8(isolate, worker->error_message.c_str()));
    Local<Value> argv[] = { err };
    worker->cb->Call(Null(isolate), 1, argv);
  }
  else {
    Local<Value> argv[] = {
      Null(isolate)
    };

    worker->cb->Call(Null(isolate), 1, argv);
  }

  worker->cb.Clear();

  delete worker;
}

//
// ### split_data
// Split data among received NNs
// ```
// @nns       {NN[]} an array of NNs
// @no_nns    {int} the number of NNs
// @step      {int} the step number
// @step_size {int} the step size
// @train_in  {vector< vector <double> >} the training set ins
// @train_out {vector< vector <double> >} the training set outs
//
// @return    {int} the total number of points added
// ```
//
int MT_NN::split_data(NN** nns, int no_nns, int step, int step_size,
                      vector< vector<double> > train_in,
                      vector< vector<double> > train_out)
{
  unsigned int total = 0;
  int max_to_insert = std::min((step + 1) * no_nns * step_size,
                          (int)train_in.size());
  int to_insert = 0;

  while(to_insert < max_to_insert - 1) {
    if(step_size > (int)train_in.size()) {
      to_insert = total % train_in.size();
    }
    else {
      to_insert = total + ((step * no_nns * step_size) % train_in.size());
    }

    if((int)train_in.size() > to_insert && (int)train_out.size() > to_insert) {
      nns[total % no_nns]->train_set_add(train_in[to_insert],
                                         train_out[to_insert]);
    }
    else {
      cout << "Wrong training set: IN " << train_in.size() <<
        " OUT " << train_out.size();
      /* Stop here */
      total = train_in.size();
    }

    total ++;
  }

  return total;
}

//
// ### learn
// Runs the NN learn step
// ```
// @arg {} the NN to train
// ```
//
void MT_NN::learn(void *arg) {
  LearnWorker *worker = (LearnWorker*)arg;

  NN *nn = worker->nn;

  worker->error = nn->learn_step();
}
