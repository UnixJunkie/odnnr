
module CLI = Minicli.CLI
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
  let argc, args = CLI.init () in
  let train_portion_def = 0.8 in
  let show_help = CLI.get_set_bool ["-h";"--help"] args in
  if argc = 1 || show_help then
    begin
      eprintf "usage:\n\
               %s\n  \
               [--train <train.txt>]: training set\n  \
               [-p <float>]: train portion; default=%f\n  \
               [--seed <int>]: RNG seed\n  \
               [--test <test.txt>]: test set\n  \
               [--epochs <int>]: optimal number of training epochs\n  \
               [-np <int>]: max CPU cores\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-s|--save <filename>]: save model to file\n  \
               [-l|--load <filename>]: restore model from file\n  \
               [--loss {RMSE|MSE|MAE}]: minimized loss and perf. metric\n  \
               (default=RMSE)\n  \
               [--optim {SGD|RMS|Ada|AdaD|AdaG|AdaM|Nada|Ftrl}]: optimizer\n  \
               (default=RMS)\n  \
               [--active {relu|sigmo}]: hidden layer activation function\n  \
               [-o <filename>]: predictions output file\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0) train_portion_def;
      exit 1
    end;
  let verbose = CLI.get_set_bool ["-v"] args in
  let no_plot = CLI.get_set_bool ["--no-plot"] args in
  let maybe_train_fn = CLI.get_string_opt ["--train"] args in
  let maybe_test_fn = CLI.get_string_opt ["--test"] args in
  let nb_epochs = CLI.get_int ["--epochs"] args in
  let loss =
    let loss_str = CLI.get_string_def ["--loss"] args "MSE" in
    DNNR.metric_of_string loss_str in
  let optimizer =
    let optim_str = CLI.get_string_def ["--optim"] args "RMS" in
    DNNR.optimizer_of_string optim_str in
  let activation =
    let activation_str = CLI.get_string_def ["--active"] args "relu" in
    DNNR.activation_of_string activation_str in
  CLI.finalize ();
  match maybe_train_fn, maybe_test_fn with
  | (Some train_fn, Some test_fn) ->
    begin
      let model_fn =
        DNNR.(train verbose
                optimizer
                loss (* loss *)
                loss (* metric *)
                [(64, activation); (64, activation)] (* architecture *)
                nb_epochs
                train_fn
             ) in
      Log.info "trained_model: %s" model_fn;
      let actual = extract_values verbose test_fn in
      let preds = DNNR.predict verbose model_fn test_fn in
      let test_R2 = Cpm.RegrStats.r2 actual preds in
      (if not no_plot then
         let title = sprintf "DNN model fit; R2=%.2f" test_R2 in
         Gnuplot.regr_plot title actual preds
      );
      Log.info "testR2: %f" test_R2
    end
  | _ -> failwith "not implemented yet"

let () = main ()
