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

#include <node.h>
#include <v8.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <uv.h>

using namespace v8;
using namespace node;
using namespace std;

//
// ## NN Class
//
class NN : public ObjectWrap {
public:
  NN(vector<int> &, double, double, double);
  NN(std::string &);
  NN(NN const&);
  ~NN();

  /**************************************************************************/
  /*                               METHODS                                  */
  /**************************************************************************/

  //
  // ### fRand
  // ```
  // @min {double} lower bound
  // @max {double} upper bound
  // ```
  //
  static double fRand(double, double);

  //
  // ### run
  // ```
  // @in {vector<double>} input vector
  // ```
  //
  vector<double> run(vector<double> &);

  //
  // ### train_set_add
  // ```
  // @in         {vector<double>} input vector
  // @out        {vector<double>} result vector to learn on
  // ```
  //
  void train_set_add(vector<double> &,
                     vector<double> &);

  //
  // ### train_set_clear
  //
  void train_set_clear();

  //
  // ### train
  // Monothreaded train
  // ```
  // @error      {double} target error
  // @iterations {int} max number of iterations
  // ```
  //
  void train(double, int);

  //
  // ### mt_train
  // Multithreaded train
  // ```
  // @error      {double} target error
  // @iterations {int} max number of iterations
  // @step_size  {int} size of training set by step
  // @n_threads  {int} the number of threads to use
  // ```
  //
  void mt_train(double, int, int, int);

  //
  // ### learn
  // ```
  // @in {vector<double>} input vector
  // @out {vector<double>} result vector
  // ```
  //
  vector<double> learn(vector<double> &,
                       vector<double> &);

  //
  // ### learn_step
  //
  double learn_step();

  //
  // ### to_string
  //
  std::string to_string();

  //
  // ### get_state
  //
  std::string get_state(bool);

  //
  // ### set_log
  //
  void set_log(bool);

  /**************************************************************************/
  /*                                BINDINGS                                */
  /**************************************************************************/

  static void Init(Handle<Object> exports);

private:
  //
  // ### bindings
  //
  static Handle<Value> New(const Arguments& args);
  static Handle<Value> TrainSetAdd(const Arguments& args);
  static Handle<Value> Train(const Arguments& args);
  static Handle<Value> MTTrain(const Arguments& args);
  static Handle<Value> Run(const Arguments& args);
  static Handle<Value> ToString(const Arguments& args);
  static Handle<Value> GetState(const Arguments& args);
  static Handle<Value> SetLog(const Arguments& args);

  //
  // ### Operators
  //
  NN& operator+=(NN const&);
  NN& operator-=(NN const&);
  NN& operator/=(int const&);

  /**************************************************************************/
  /*                              MEMBERS                                   */
  /**************************************************************************/

  vector< vector< vector<double> > > W_;         /* weights */
  vector< vector< vector<double> > > dW_;        /* changes */
  vector< vector<double> >           B_;         /* bias weights */

  vector< vector<double> >           D_;         /* deltas */
  vector< vector<double> >           sum_;       /* incoming sums */
  vector< vector<double> >           val_;       /* values */

  vector<int>                        layers_;    /* layers structure */
  int                                L_;         /* layers count */
  long                               op_count_;  /* op count */

  double                             alpha_;     /* learning rate */
  double                             beta_;      /* momentum */
  double                             bias_;      /* bias value */

  vector< vector<double> >           train_in_;  /* training set input */
  vector< vector<double> >           train_out_; /* training set out */

  bool                               log_;       /* Whether to log outputs */
};


/******************************************************************************/
/*                           MULTITHREADING HELPERS                           */
/******************************************************************************/

namespace MT_NN {
  //
  // ### Functions
  //
  void train_start(uv_work_t* req);
  void train_done(uv_work_t* req, int status);

  int split_data(NN**, int, int, int,
                 vector< vector<double> >,
                 vector< vector<double> >);
  void learn(void *arg);

  //
  // ## TrainWorker struct
  //
  struct TrainWorker {
    uv_work_t request;
    Persistent<Function> cb;
    string error_message;

    double target_error;
    int iterations;
    int step_size;
    int threads;

    NN* nn;
  };

  //
  // ## LearnWorker struct
  //
  struct LearnWorker {
    NN* nn;
    double error;
  };
};
