

def result(request):

    process_runningtime = time()

    inputDic = {"target_corp": request.GET['selectedCom'],
                "input_model": request.GET['selectedModel'],
                "pred_start": request.GET['예측날짜'].replace("-", ""),
                "target_ntime": int(request.GET['예측기간']),
                "train_period": int(request.GET['훈련기간'])}

    target_timegap = inputDic["target_ntime"]

    # Get Data & Modeling
    # 분석할 date 변수 지정
    start_date = (datetime.strptime(inputDic["pred_start"], "%Y%m%d") - BDay(inputDic["train_period"])).strftime("%Y%m%d")
    end_date = inputDic["pred_start"]
    business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

    tickers = []
    tickers += stock.get_market_ticker_list(market="KOSPI")
    tickers += stock.get_market_ticker_list(market="KOSDAQ")

    ticker_names = []
    for i in tickers:
        ticker_names.append(stock.get_market_ticker_name(i))
    stock_list = dataframe(columns=["종목명", "종목코드"])
    stock_list["종목명"] = ticker_names
    stock_list["종목코드"] = tickers
    if stock_list.isna().sum().sum() != 0:
        raise Http404("stock list has NA values")

    stock_list.set_index("종목명", inplace=True)
    selected_codes = stock_list.index[stock_list.index == inputDic["target_corp"]].to_list()
    stock_list = stock_list.loc[selected_codes]["종목코드"]

    stock_dic = dict.fromkeys(selected_codes)
    error_list = []
    corr_list = []
    metric_days = 14
    cat_vars = []
    bin_vars = []
    cat_vars.append("weekday")
    cat_vars.append("weeknum")
    bin_vars.append("mfi_signal")

    # ==== selected feature =====
    selected_features = ["date", "close", "kospi", "obv", "trading_amount", "mfi_signal"]
    logtrans_vec = ["close", "kospi", "trading_amount"]
    pvalue_check = series(0, index=selected_features)

    dataloading_runningtime = time()
    for stock_name, stock_code in stock_list.items():
        print("=====", stock_name, "=====")
        # 종목 주가 데이터 로드
        try:
            stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
            stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
            investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
            kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
            # sleep(0.5)

            stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            investor_df.columns = ["Date", "inst", "fore"]
            kospi_df.columns = ["Date", "kospi"]
            # 영업일과 주가 정보를 outer 조인
            train_x = pd.merge(business_days, stock_df, how='left', on="Date")
            train_x = pd.merge(train_x, investor_df, how='left', on="Date")
            train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
            # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
            train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
        except:
            stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
            stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
            sleep(0.5)
            # investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
            # sleep(0.5)
            # kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
            # sleep(0.5)

            stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            # investor_df.columns = ["Date", "inst", "fore"]
            # kospi_df.columns = ["Date", "kospi"]
            # 영업일과 주가 정보를 outer 조인
            train_x = pd.merge(business_days, stock_df, how='left', on="Date")
            # train_x = pd.merge(train_x, investor_df, how='left', on="Date")
            # train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
            # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
            train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)

        # ===== feature engineering =====
        # 거래대금 파생변수 추가
        train_x['trading_amount'] = train_x["Close"] * train_x["Volume"]

        # OBV 파생변수 추가
        # 매수 신호: obv > obv_ema
        # 매도 신호: obv < obv_ema
        obv = [0]
        for i in range(1, len(train_x.Close)):
            if train_x.Close[i] >= train_x.Close[i - 1]:
                obv.append(obv[-1] + train_x.Volume[i])
            elif train_x.Close[i] < train_x.Close[i - 1]:
                obv.append(obv[-1] - train_x.Volume[i])
            # else:
            #     obv.append(obv[-1])
        train_x['obv'] = obv
        train_x['obv'][0] = nan

        # MFI 파생변수 추가
        # MFI = 100 - (100 / 1 + MFR)
        # MFR = 14일간의 양의 MF / 14일간의 음의 MF
        # MF = 거래량 * (당일고가 + 당일저가 + 당일종가) / 3
        # MF 컬럼 만들기
        train_x["mf"] = train_x["Volume"] * ((train_x["High"] + train_x["Low"] + train_x["Close"]) / 3)
        # 양의 MF와 음의 MF 표기 컬럼 만들기
        p_n = []
        for i in range(len(train_x['mf'])):
            if i == 0:
                p_n.append(nan)
            else:
                if train_x['mf'][i] >= train_x['mf'][i - 1]:
                    p_n.append('p')
                else:
                    p_n.append('n')
        train_x['p_n'] = p_n
        # 14일간 양의 MF/ 14일간 음의 MF 계산하여 컬럼 만들기
        mfr = []
        for i in range(len(train_x['mf'])):
            if i < metric_days - 1:
                mfr.append(nan)
            else:
                train_x_ = train_x.iloc[(i - metric_days + 1):i]
                a = (sum(train_x_['mf'][train_x['p_n'] == 'p']) + 1) / (sum(train_x_['mf'][train_x['p_n'] == 'n']) + 10)
                mfr.append(a)
        train_x['mfr'] = mfr
        # 최종 MFI 컬럼 만들기
        train_x['mfi'] = 100 - (100 / (1 + train_x['mfr']))
        train_x["mfi_signal"] = train_x['mfi'].apply(lambda x: "buy" if x > 50 else "sell")

        # 지표계산을 위해 쓰인 컬럼 drop
        train_x.drop(["mf", "p_n", "mfr", "Open", "High", "Low"], inplace=True, axis=1)

        train_x = train_x.dropna()
        train_x.reset_index(drop=True, inplace=True)
        print("NA values --->", train_x.isna().sum().sum())

        # create target list
        target_list = []
        target_list.append(train_x["Close"])
        for i in range(1,target_timegap+1,1):
            target_list.append(train_x["Close"].shift(-i))
        for idx, value in enumerate(target_list):
            value.name = "target_shift" + str(idx)

        # 컬럼이름 소문자 변환 및 정렬
        train_x.columns = train_x.columns.str.lower()
        train_x = pd.concat([train_x[["date"]], train_x.iloc[:, 1:].sort_index(axis=1)], axis=1)

        # # <visualization>
        # # 시각화용 데이터프레임 생성
        # train_bi = pd.concat([target_list[timeunit_gap_forviz], train_x], axis=1)[:-timeunit_gap_forviz]
        #
        # # 기업 평균 상관관계를 측정하기 위한 연산
        # corr_obj = train_bi.corr().round(3)
        # corr_rows = corr_obj.index.tolist()
        # corr_cols = corr_obj.columns.tolist()
        # corr_list.append(corr_obj.to_numpy().round(3)[..., np.newaxis])

        # <feature selection>
        if len(selected_features) != 0:
            train_x = train_x[np.intersect1d(train_x.columns, selected_features)]

        # <feature scaling>
        # log transform
        for i in logtrans_vec:
            if i in train_x.columns:
                train_x[i] = train_x[i].apply(np.log1p)

        # onehot encoding
        onehot_encoder = MyOneHotEncoder()
        train_x = onehot_encoder.fit_transform(train_x, cat_vars + bin_vars)

        stock_dic[stock_name]["df"] = train_x.copy()
        stock_dic[stock_name]["target_list"] = target_list.copy()
    dataloading_runningtime = time() - dataloading_runningtime

    # ===== Automation Predict =====
    # validation data evaluation
    model_names = ["Linear", "ElasticNet", "KNN", "XGB_GBT",
                   "LGB_RandomForest", "LGB_GOSS", "ARIMA", "LSTM"]


    seqLength = 5
    output_str = ""
    output_list = []

    # 데이터를 저장할 변수 설정
    total_perf = None
    for stock_name, stock_data in stock_dic.items():
        stock_data["perf_list"] = dict.fromkeys(model_names)
        stock_data["pred_list"] = dict.fromkeys(model_names)
        total_perf = dict.fromkeys(model_names)
        for i in model_names:
            stock_data["perf_list"][i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            stock_data["pred_list"][i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            total_perf[i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            for j in total_perf[i].keys():
                total_perf[i][j] = series(0, index=["MAE", "MAPE", "NMAE", "RMSE", "NRMSE", "R2", "Running_Time"])

    fit_runningtime = time()
    for time_ngap in range(1, target_timegap + 1):
        print(F"=== Target on N+{time_ngap} ===")
        # time_ngap = 3
        for stock_name, stock_data in stock_dic.items():
            # remove date
            # break

            test_x = stock_data["df"].iloc[-1:]
            test_x_lstm = stock_data["df"].iloc[-seqLength:]

            full_x = stock_data["df"][:-time_ngap]
            full_y = stock_data["target_list"][time_ngap][:-time_ngap]
            full_x_lstm = stock_data["df"][:-time_ngap]
            full_y_lstm = full_y[seqLength - 1:]
            arima_full = stock_data["target_list"][0]

            # create dataset for fitting
            data_forfit = stock_data["df"][:-time_ngap]
            target_forfit = stock_data["target_list"][time_ngap][:-time_ngap]

            val_x = data_forfit.iloc[-1:]
            val_y = target_forfit.iloc[-1:]
            val_x_lstm = data_forfit.iloc[-seqLength:]
            val_y_lstm = target_forfit.iloc[-1:]

            train_x = data_forfit[:-time_ngap]
            train_y = target_forfit[:-time_ngap]
            train_x_lstm = data_forfit[:-time_ngap]
            train_y_lstm = train_y[seqLength - 1:]
            arima_train = stock_data["target_list"][0][:-time_ngap][:-time_ngap]



            full_x.drop("date", axis=1, inplace=True)
            full_x_lstm.drop("date", axis=1, inplace=True)
            train_x.drop("date", axis=1, inplace=True)
            train_x_lstm.drop("date", axis=1, inplace=True)
            val_x.drop("date", axis=1, inplace=True)
            val_x_lstm.drop("date", axis=1, inplace=True)
            test_x.drop("date", axis=1, inplace=True)
            test_x_lstm.drop("date", axis=1, inplace=True)

            if inputDic["input_model"] == "Linear":
                # <선형회귀>
                tmp_runtime = time()
                print("Linear Regression on", stock_name)
                # evaludation on validation set
                model = doLinear(train_x, train_y, val_x, val_y)
                print(model["performance"])
                stock_data["perf_list"]["Linear"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])

                # prediction on test set
                model = doLinear(full_x, full_y, test_x, None)
                stock_data["pred_list"]["Linear"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["Linear"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["Linear"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["Linear"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "ElasticNet":
                # <엘라스틱넷>
                tmp_runtime = time()
                print("ElasticNet on", stock_name)
                # evaludation on validation set
                model = doElasticNet(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
                print(model["performance"])
                stock_data["perf_list"]["ElasticNet"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                # prediction on test set
                model = doElasticNet(full_x, full_y, test_x, None, kfolds=kfolds_spliter, tuningMode=False,
                                     alpha=model["best_params"]["alpha"], l1_ratio=model["best_params"]["l1_ratio"])
                stock_data["pred_list"]["ElasticNet"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["ElasticNet"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["ElasticNet"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["ElasticNet"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "KNN":
                # <KNN>
                tmp_runtime = time()
                print("KNN on", stock_name)
                # evaludation on validation set
                model = doKNN(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
                print(model["performance"])
                stock_data["perf_list"]["KNN"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                # prediction on test set
                model = doKNN(full_x, full_y, test_x, None, k=model["best_params"]["n_neighbors"],
                              tuningMode=False, kfolds=kfolds_spliter)
                stock_data["pred_list"]["KNN"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["KNN"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["KNN"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["KNN"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "XGB_GBT":
                # <XGBoost>
                tmp_runtime = time()
                print("XGB_GBT on", stock_name)
                # evaludation on validation set
                model = doXGB(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter,
                              ntrees=1000, eta=1e-2,
                              depthSeq=[4], subsampleSeq=[0.8], colsampleSeq=[1.0], gammaSeq=[0.0])
                print(model["performance"])
                stock_data["perf_list"]["XGB_GBT"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                print(model["best_params"])
                # prediction on test set
                model = doXGB(full_x, full_y, test_x, None, tuningMode=False,
                              ntrees=model["best_params"]["best_trees"],
                              depthSeq=model["best_params"]["max_depth"],
                              mcwSeq=model["best_params"]["min_child_weight"],
                              l2Seq=model["best_params"]["reg_lambda"],
                              gammaSeq=model["best_params"]["gamma"],
                              subsampleSeq=model["best_params"]["subsample"],
                              colsampleSeq=model["best_params"]["colsample_bytree"])
                stock_data["pred_list"]["XGB_GBT"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["XGB_GBT"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["XGB_GBT"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["XGB_GBT"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "LGB_RandomForest":
                # <LightGBM 랜덤포레스트>
                tmp_runtime = time()
                # GridSearchCV의 param_grid 설정
                print("LGB_RandomForest on", stock_name)
                params = {
                    'learning_rate': [0.01],
                    'num_leaves': [2 ** i - 1 for i in [4, 6]],
                    'n_estimators': [100]
                }

                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_jobs=None, random_state=321)
                grid = GridTuner(estimator=model, param_grid=params,
                                 n_jobs=multiprocessing.cpu_count(),
                                 refit=False, cv=kfolds_spliter)
                grid.fit(train_x, train_y)

                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_estimators=grid.best_params_["n_estimators"],
                                          num_leaves=grid.best_params_["num_leaves"],
                                          learning_rate=grid.best_params_["learning_rate"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=20, eval_metric='rmse', verbose=0)
                pred = model.predict(val_x)
                print(grid.best_params_)
                print("best iteration --->", model.best_iteration_)

                # recode performance
                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                model_perf = {"MAE": tmp_mae,
                              "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                              "NMAE": tmp_mae / val_y.abs().mean(),
                              "RMSE": tmp_rmse,
                              "NRMSE": tmp_rmse / val_y.abs().mean(),
                              "R2": metrics.r2_score(val_y, pred)}
                print(model_perf)
                stock_data["perf_list"]["LGB_RandomForest"][time_ngap] = model_perf

                # prediction on test data
                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_estimators=model.best_iteration_,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          learning_rate=grid.best_params_["learning_rate"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(full_x, full_y, verbose=0)
                pred = model.predict(test_x)
                stock_data["pred_list"]["LGB_RandomForest"][time_ngap] = pred
                # recode running time
                tmp_runtime = time() - tmp_runtime
                print(tmp_runtime)
                total_perf["LGB_RandomForest"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["LGB_RandomForest"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LGB_RandomForest"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "LGB_GOSS":
                # <LightGBM Gradient-based One-Side Sampling>
                tmp_runtime = time()
                # GridSearchCV의 param_grid 설정
                print("LGB_GOSS on", stock_name)
                params = {
                    'learning_rate': [5e-4],
                    'n_estimators': [2000],
                    'num_leaves': [2 ** i - 1 for i in [4, 6]],
                    'reg_lambda': [0.1, 1.0, 5.0],
                    'min_child_samples': [5, 10, 20]
                }

                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression",
                                          subsample=0.8, n_jobs=None, random_state=321)
                grid = GridTuner(estimator=model, param_grid=params,
                                 n_jobs=multiprocessing.cpu_count(),
                                 refit=False, cv=kfolds_spliter)
                grid.fit(train_x, train_y)

                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                          n_estimators=5000,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          reg_lambda=grid.best_params_["reg_lambda"],
                                          min_child_samples=grid.best_params_["min_child_samples"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=500, eval_metric='rmse', verbose=0)
                pred = model.predict(val_x)
                print(grid.best_params_)
                print("best iteration --->", model.best_iteration_)

                # recode performance
                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                model_perf = {"MAE": tmp_mae,
                              "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                              "NMAE": tmp_mae / val_y.abs().mean(),
                              "RMSE": tmp_rmse,
                              "NRMSE": tmp_rmse / val_y.abs().mean(),
                              "R2": metrics.r2_score(val_y, pred)}
                print(model_perf)
                stock_data["perf_list"]["LGB_GOSS"][time_ngap] = model_perf

                # prediction on test data
                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                          n_estimators=model.best_iteration_,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          reg_lambda=grid.best_params_["reg_lambda"],
                                          min_child_samples=grid.best_params_["min_child_samples"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(full_x, full_y, verbose=0)
                pred = model.predict(test_x)
                stock_data["pred_list"]["LGB_GOSS"][time_ngap] = pred
                # recode running time
                tmp_runtime = time() - tmp_runtime
                print(tmp_runtime)
                total_perf["LGB_GOSS"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["LGB_GOSS"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LGB_GOSS"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "ARIMA":
                # <ARIMA>
                tmp_runtime = time()
                print("ARIMA on", stock_name)
                # order=(p: Auto regressive, q: Difference, d: Moving average)
                # 일반적 하이퍼파라미터 공식
                # 1. p + q < 2
                # 2. p * q = 0
                # 근거 : 실제로 대부분의 시계열 자료에서는 하나의 경향만을 강하게 띄기 때문 (p 또는 q 둘중 하나는 0)
                model = ARIMA(arima_train, order=(1, 2, 0))
                model_fit = model.fit()
                pred = array([model_fit.forecast(target_timegap).iloc[-1]])

                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                tmp_perf = {"MAE": tmp_mae,
                            "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                            "NMAE": tmp_mae / val_y.abs().mean(),
                            "RMSE": tmp_rmse,
                            "NRMSE": tmp_rmse / val_y.abs().mean(),
                            "R2": metrics.r2_score(val_y, pred)}
                print(tmp_perf)
                stock_data["perf_list"]["ARIMA"][target_timegap] = tmp_perf

                # prediction on test data
                model = ARIMA(arima_full, order=(1, 2, 0))
                model_fit = model.fit()

                for idx, value in enumerate(model_fit.forecast(target_timegap)):
                    stock_data["pred_list"]["ARIMA"][idx+1] = value

                tmp_runtime = time() - tmp_runtime
                total_perf["ARIMA"][target_timegap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))
                for idx, value in stock_data["pred_list"]["ARIMA"].items():
                    output_list.append(value)
                    output_str += str(idx) + " ---> " + str(value) + "\n"
            elif inputDic["input_model"] == "LSTM":
                pass
                # <LSTM>
                tmp_runtime = time()
                print("LSTM on", stock_name)
                model = doMLP(train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, mlpName="MLP_LSTM_V1",
                              hiddenLayers=64, epochs=100, batch_size=4, seqLength=seqLength, model_export=True)
                print(model["performance"])
                stock_data["perf_list"]["LSTM"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])

                model = doMLP(full_x_lstm, full_y_lstm, test_x_lstm, None,
                              seqLength=seqLength, preTrained=model["model"])
                stock_data["pred_list"]["LSTM"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["LSTM"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LSTM"][time_ngap][0]) + "\n"
            else:
                print("WARNNING : Unknown model")
        print(output_str)
        if inputDic["input_model"] == "ARIMA":
            break