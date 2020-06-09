
module CLI = Minicli.CLI
module DNNR = Odnnr.DNNR
module Fn = Filename
module L = BatList
module Utls = Odnnr.Utls
module Log = Dolog.Log
module Parmap = Parany.Parmap

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

let train_test_dump csv_header train test =
  let train_fn = Fn.temp_file "odnnr_train_" ".csv" in
  let test_fn = Fn.temp_file "odnnr_test_" ".csv" in
  Utls.lines_to_file train_fn (csv_header :: train);
  Utls.lines_to_file test_fn (csv_header :: test);
  (train_fn, test_fn)

let shuffle_then_cut seed p train_fn =
  match Utls.lines_of_file train_fn with
  | [] | [_] -> assert(false) (* no lines or header line only?! *)
  | (csv_header :: csv_payload) ->
    let rng = BatRandom.State.make [|seed|] in
    let rand_lines = L.shuffle ~state:rng csv_payload in
    let train, test = Utls.train_test_split p rand_lines in
    train_test_dump csv_header train test

let shuffle_then_nfolds seed n train_fn =
  match Utls.lines_of_file train_fn with
  | [] | [_] -> assert(false) (* no lines or header line only?! *)
  | (csv_header :: csv_payload) ->
    let rng = BatRandom.State.make [|seed|] in
    let rand_lines = L.shuffle ~state:rng csv_payload in
    let train_tests = Utls.cv_folds n rand_lines in
    L.rev_map (fun (x, y) -> train_test_dump csv_header x y) train_tests

let train_test verbose no_plot
    optimizer loss hidden_layers nb_epochs train_fn test_fn =
  let model_fn =
    DNNR.(train verbose
            optimizer
            loss (* loss *)
            loss (* metric *)
            hidden_layers
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
  let arch_str = DNNR.string_of_layers hidden_layers in
  Log.info "%s R2_te: %.3f" arch_str test_R2;
  test_R2

let early_stop verbose optimizer loss hidden_layers
    max_epochs delta_epochs patience train_fn test_fn =
  let actual = extract_values verbose test_fn in
  let arch_str = DNNR.string_of_layers hidden_layers in
  let model_fn = Fn.temp_file "odnnr_model_" ".bin" in
  (DNNR.early_stop_init
     verbose
     optimizer
     loss
     loss
     hidden_layers
     delta_epochs
     train_fn
     model_fn);
  let init_R2 =
    let preds = DNNR.predict verbose model_fn test_fn in
    Cpm.RegrStats.r2 actual preds in
  Log.info "%s %d R2_te: %.3f" arch_str delta_epochs init_R2;
  let rec loop best_R2 best_epochs curr_epochs nb_fails =
    if curr_epochs >= max_epochs then
      (Log.error "Model.early_stop: max epochs reached";
       (model_fn, best_R2, best_epochs))
    else if nb_fails = patience then
      (Log.info "Model.early_stop: patience reached";
       (model_fn, best_R2, best_epochs))
    else
      begin
        DNNR.early_stop_continue verbose delta_epochs model_fn;
        let curr_R2 =
          let preds = DNNR.predict verbose model_fn test_fn in
          Cpm.RegrStats.r2 actual preds in
        Log.info "%s %d R2_te: %.3f" arch_str curr_epochs curr_R2;
        if curr_R2 > best_R2 then
          loop curr_R2 curr_epochs (curr_epochs + delta_epochs) 0
        else (* curr_R2 <= best_R2 *)
          loop best_R2 best_epochs (curr_epochs + delta_epochs) (nb_fails + 1)
      end in
  loop init_R2 delta_epochs delta_epochs 0

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
               [--epochs <int>]: optimal/max number of training epochs\n  \
               [-np <int>]: max CPU cores\n  \
               [--early-stop]: early stopping epochs scan\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [-s|--save <filename>]: save model to file\n  \
               [-l|--load <filename>]: restore model from file\n  \
               [--loss {RMSE|MSE|MAE}]: minimized loss and perf. metric\n  \
               (default=RMSE)\n  \
               [--optim {SGD|RMS|Ada|AdaD|AdaG|AdaM|Nada|Ftrl}]: optimizer\n  \
               (default=RMS)\n  \
               [--active {relu|sigmo}]: hidden layer activation function\n  \
               [--arch {<int>/<int>/...}]: size of each hidden layer\n  \
               [-o <filename>]: predictions output file\n  \
               [--no-plot]: don't call gnuplot\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this message\n"
        Sys.argv.(0) train_portion_def;
      exit 1
    end;
  let verbose = CLI.get_set_bool ["-v"] args in
  let epochs_scan = CLI.get_set_bool ["--early-stop"] args in
  let ncores = CLI.get_int_def ["-np";"--nprocs"] args 1 in
  let seed = match CLI.get_int_opt ["-s";"--seed"] args with
    | Some s -> s (* reproducible *)
    | None -> (* random *)
      let () = Random.self_init () in
      Random.int 0x3FFFFFFF (* 0x3FFFFFFF = 2^30 - 1 *) in
  let no_plot = CLI.get_set_bool ["--no-plot"] args in
  let maybe_train_fn = CLI.get_string_opt ["--train"] args in
  let maybe_test_fn = CLI.get_string_opt ["--test"] args in
  let nb_epochs = CLI.get_int ["--epochs"] args in
  let nfolds = CLI.get_int_def ["--NxCV"] args 1 in
  let train_portion = CLI.get_float_def ["-p"] args 0.8 in
  let loss =
    let loss_str = CLI.get_string_def ["--loss"] args "MSE" in
    DNNR.metric_of_string loss_str in
  let optimizer =
    let optim_str = CLI.get_string_def ["--optim"] args "RMS" in
    DNNR.optimizer_of_string optim_str in
  let activation =
    let activation_str = CLI.get_string_def ["--active"] args "relu" in
    DNNR.activation_of_string activation_str in
  let arch_str = CLI.get_string_def ["--arch"] args "64/64" in
  let hidden_layers = DNNR.layers_of_string activation arch_str in
  CLI.finalize ();
  match maybe_train_fn, maybe_test_fn with
  | (None, None) -> failwith "provide --train and/or --test"
  | (None, Some _test) -> failwith "only --test: not implemented yet"
  | (Some train_fn, Some test_fn) ->
    ignore(train_test verbose no_plot
             optimizer loss hidden_layers nb_epochs train_fn test_fn)
  | (Some train_fn', None) ->
    if nfolds > 1 then
      begin
        (* NxCV *)
        Log.info "shuffle -> %dxCV" nfolds;
        let train_test_fns = shuffle_then_nfolds seed nfolds train_fn' in
        let r2s =
          (* we core pin so that the R processes should be confined
             to a single core *)
          Parmap.parmap ~core_pin:true ncores (fun (train_fn, test_fn) ->
              train_test verbose no_plot
                optimizer loss hidden_layers nb_epochs train_fn test_fn
            ) train_test_fns in
        let r2_avg = Utls.favg r2s in
        let arch_str = DNNR.string_of_layers hidden_layers in
        Log.info "%s avg(R2_te): %.3f" arch_str r2_avg
      end
    else
      begin
        (* train/test split *)
        Log.info "shuffle -> train/test split (p=%.2f)" train_portion;
        let train_fn, test_fn = shuffle_then_cut seed train_portion train_fn' in
        if epochs_scan then
          let model_fn, best_R2, best_epochs =
            early_stop verbose optimizer loss hidden_layers
              nb_epochs 1 5 train_fn test_fn in
          Log.info "model_fn: %s best_R2: %.3f epochs: %d"
            model_fn best_R2 best_epochs
        else
          ignore(train_test verbose no_plot
                   optimizer loss hidden_layers nb_epochs train_fn test_fn)
      end

let () = main ()
