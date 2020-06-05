
module DNNR = Odnnr.DNNR
module Fn = Filename
module Utls = Odnnr.Utls
module Log = Dolog.Log

open Printf

let extract_values verbose fn =
  let actual_fn = Fn.temp_file "odnnr_test_" ".txt" in
  (* NR > 1: skip CSV header line *)
  let cmd = sprintf "awk '(NR > 1){print $1}' %s > %s" fn actual_fn in
  Utls.run_command verbose cmd;
  let actual = Utls.float_list_of_file actual_fn in
  (* filesystem cleanup *)
  (if not verbose then Sys.remove actual_fn);
  actual

let main () =
  Log.(set_log_level DEBUG);
  Log.color_on ();
  let no_plot = false in
  let model_fn =
    DNNR.(train
            true
            RMSprop
            MSE
            MSE
            [(64, Relu); (64, Relu)]
            50
            "data/Boston_regr_train.csv"
         ) in
  Log.info "trained_model: %s" model_fn;
  let actual = extract_values true "data/Boston_regr_test.csv" in
  let preds = DNNR.predict true model_fn "data/Boston_regr_test.csv" in
  let test_R2 = Cpm.RegrStats.r2 actual preds in
  (if not no_plot then
     let title = sprintf "DNN model fit; R2=%.2f" test_R2 in
     Gnuplot.regr_plot title actual preds
  );
  Log.info "testR2: %f" test_R2

let () = main ()
