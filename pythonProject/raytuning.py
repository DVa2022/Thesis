if __name__ == '__main__':

    search_space = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
    }
    tuner = tune.Tuner(
        main,
        tune_config=tune.TuneConfig(
            num_samples=3,
            scheduler=ASHAScheduler(metric="Average Precision", mode="max"),
        ),
        param_space=search_space,
    )

    # space = {
    #     "lr": hp.loguniform("lr", -10, -1),
    #     "momentum": hp.uniform("momentum", 0.1, 0.9),
    # }
    #
    # hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")
    #
    # tuner = tune.Tuner(
    #     train_mnist,
    #     tune_config=tune.TuneConfig(
    #         num_samples=10,
    #         search_alg=hyperopt_search,
    #     ),
    # )

    results = tuner.fit()

    # To enable GPUs, use this instead:
    # analysis = tune.run(
    #     train_mnist, config=search_space, resources_per_trial={'gpu': 1})

    dfs = {result.path: result.metrics_dataframe for result in results}
    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.mean_accuracy.plot(ax=ax, legend=False)

    logdir = results.get_best_result("Average Precision", mode="max").path
    state_dict = torch.load(os.path.join(logdir, "model.pth"))

    # getting the best model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()