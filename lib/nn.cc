#include "nn.hh"

#include <sstream>

using namespace v8;
using namespace node;
using namespace std;


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
        NN* nns[thread];
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
        uv_thread_t nns_ids[n_thread];
        MT_NN::LearnWorker *workers[n_thread];
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
std::string NN::get_state()
{
  ostringstream oss;

  oss << (int)layers_.size();

  for(int i = 0; i < (int)layers_.size(); i++) {
    oss << " " << layers_[i];
  }

  for(int l = 0; l < L_; l++) {
    for(int i = 0; i < layers_[l]; i++) {
      oss << " " << val_[l][i];
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
Handle<Value> NN::ToString(const Arguments& args) {
  HandleScope scope;

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  /* return values */
  v8::Handle<v8::String> result = v8::String::New(nn->to_string().c_str());

  return scope.Close(result);
}

//
// ### GetState wrapper
//
Handle<Value> NN::GetState(const Arguments& args) {
  HandleScope scope;

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  /* return values */
  v8::Handle<v8::String> result = v8::String::New(nn->get_state().c_str());

  return scope.Close(result);
}

//
// ### TrainSetAdd wrapper
//
Handle<Value> NN::TrainSetAdd(const Arguments& args) {
  HandleScope scope;

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  if(!args[0]->IsArray()) {
    ThrowException(
      Exception::TypeError(
        String::New("Training `in` values expected as argument 0")));
    return scope.Close(Undefined());
  }
  if(!args[1]->IsArray()) {
    ThrowException(
      Exception::TypeError(
        String::New("Training `out` values expected as argument 0")));
    return scope.Close(Undefined());
  }

  /* training set extraction */
  Local<Array> in = Array::Cast(*args[0]);
  Local<Array> out = Array::Cast(*args[1]);

  vector<double> input(in->Length());
  vector<double> output(out->Length());

  for(unsigned int i = 0; i < in->Length(); i ++) {
    input[i] = in->Get(Integer::New(i))->ToNumber()->Value();
  }
  for(unsigned int i = 0; i < out->Length(); i ++) {
    output[i] = out->Get(Integer::New(i))->ToNumber()->Value();
  }

  nn->train_set_add(input, output);

  return scope.Close(Undefined());
}

//
// ### Train wrapper
//
Handle<Value> NN::Train(const Arguments& args) {
  HandleScope scope;

  /* unwraping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  /* call */
  if(args[0]->IsNumber() && args[1]->IsNumber()) {
    nn->train(args[0]->ToNumber()->Value(),
              (int)args[1]->ToNumber()->Value());
  }
  else if(args[0]->IsNumber()) {
    nn->train(args[0]->ToNumber()->Value());
  }
  else {
    nn->train();
  }

  return scope.Close(Undefined());
}

//
// ### Multithread Train Wrapper
//
Handle<Value> NN::MTTrain(const Arguments& args) {
  HandleScope scope;

  double target_error = 0;
  int iterations = 0;
  int step_size = 0;
  int threads = 0;

  Local<Function> cb;

  if(args[0]->IsNumber() && args[1]->IsNumber() &&
     args[2]->IsNumber() && args[3]->IsNumber()) {
    target_error = args[0]->ToNumber()->Value();
    iterations = (int)args[1]->ToNumber()->Value();
    step_size = (int)args[2]->ToNumber()->Value();
    threads = (int)args[3]->ToNumber()->Value();

    cb = Local<Function>::Cast(args[4]);
  }
  else if(args[0]->IsNumber() && args[1]->IsNumber() && args[2]->IsNumber()) {
    target_error = args[0]->ToNumber()->Value();
    iterations = (int)args[1]->ToNumber()->Value();
    step_size = (int)args[2]->ToNumber()->Value();

    cb = Local<Function>::Cast(args[3]);
  }
  else if(args[0]->IsNumber() && args[1]->IsNumber()) {
    target_error = args[0]->ToNumber()->Value();
    iterations = (int)args[1]->ToNumber()->Value();

    cb = Local<Function>::Cast(args[2]);
  }
  else if(args[0]->IsNumber()) {
    target_error = args[0]->ToNumber()->Value();

    cb = Local<Function>::Cast(args[1]);
  }
  else {
    cb = Local<Function>::Cast(args[0]);
  }

  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  MT_NN::TrainWorker *worker = new MT_NN::TrainWorker();

  worker->request.data = worker;
  worker->cb = Persistent<Function>::New(cb);
  worker->nn = nn;

  worker->target_error = target_error;
  worker->iterations = iterations;
  worker->step_size = step_size;
  worker->threads = threads;

  uv_queue_work(uv_default_loop(), &worker->request,
                MT_NN::train_start, MT_NN::train_done);

  return scope.Close(Undefined());
}


//
// ### Run wrapper
//
Handle<Value> NN::Run(const Arguments& args) {
  HandleScope scope;

  /* unwrapping */
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  if(!args[0]->IsArray()) {
    ThrowException(
      Exception::TypeError(String::New("Input expected as argument 0")));
    return scope.Close(Undefined());
  }

  Local<Array> l = Array::Cast(*args[0]);
  vector<double> input(l->Length());

  for(unsigned int i = 0; i < l->Length(); i ++) {
    input[i] = l->Get(Integer::New(i))->ToNumber()->Value();
  }

  /* call */
  vector<double> out = nn->run(input);

  /* return values */
  v8::Handle<v8::Array> result = v8::Array::New(out.size());
  for (size_t i = 0; i < out.size(); i++)
    result->Set(Integer::New(i), Number::New(out[i]));

  return scope.Close(result);
}

//
// ### New
//
Handle<Value> NN::New(const Arguments& args) {
  HandleScope scope;
  NN* nn = NULL;

  if(args[0]->IsString()) {
    std::string str = std::string(
        *v8::String::Utf8Value(args[0]->ToString()));

    nn = new NN(str);
  }

  else if(args[0]->IsArray()) {
    Local<Array> l = Array::Cast(*args[0]);

    if(l->Length() < 2) {
      ThrowException(
        Exception::TypeError(String::New("Less than 2 Layers")));
      return scope.Close(Undefined());
    }

    vector<int> layers(l->Length());
    for(unsigned int i = 0; i < l->Length(); i++) {
      layers[i] = l->Get(Integer::New(i))->ToInteger()->Value();
    }

    nn = new NN(layers);
  }

  else {
    ThrowException(
      Exception::TypeError(String::New("Layers expected as argument 0")));
    return scope.Close(Undefined());
  }

  /* wrapping */
  nn->Wrap(args.This());
  return args.This();
}

//
// ### SetLog
//
Handle<Value> NN::SetLog(const Arguments& args) {
  HandleScope scope;
  NN* nn = ObjectWrap::Unwrap<NN>(args.This());

  bool status = false;
  if(args[0]->IsBoolean()) {
    status = args[0]->ToBoolean()->Value();
  }
  else {
    ThrowException(
      Exception::TypeError(String::New("Boolean expected as argument 0")));
    return scope.Close(Undefined());
  }

  nn->set_log(status);

  return scope.Close(Undefined());
}

/******************************************************************************/
/*                            MODULE INIT                                     */
/******************************************************************************/

//
// ### Init
//
void NN::Init(Handle<Object> exports)
{
  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(String::NewSymbol("NN"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  tpl->PrototypeTemplate()->Set(String::NewSymbol("train_set_add"),
      FunctionTemplate::New(TrainSetAdd)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("train"),
      FunctionTemplate::New(Train)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("mt_train"),
      FunctionTemplate::New(MTTrain)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("run"),
      FunctionTemplate::New(Run)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("to_string"),
      FunctionTemplate::New(ToString)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("get_state"),
      FunctionTemplate::New(GetState)->GetFunction());
  tpl->PrototypeTemplate()->Set(String::NewSymbol("set_log"),
      FunctionTemplate::New(SetLog)->GetFunction());

  Persistent<Function> constructor =
    Persistent<Function>::New(tpl->GetFunction());
  exports->Set(String::NewSymbol("NN"), constructor);
}

void InitAll(Handle<Object> exports) {
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
  HandleScope scope;
  TrainWorker* worker = static_cast<TrainWorker*>(req->data);

  if(!worker->error_message.empty()) {
    Local<Value> err = Exception::Error(
                         String::New(worker->error_message.c_str()));
    Local<Value> argv[] = { err };
    worker->cb->Call(Context::GetCurrent()->Global(), 1, argv);
  }
  else {
    Local<Value> argv[] = {
      Local<Value>::New(Null())
    };
    worker->cb->Call(Context::GetCurrent()->Global(), 1, argv);
  }

  worker->cb.Dispose();
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
  int max_to_insert = min((step + 1) * no_nns * step_size,
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
