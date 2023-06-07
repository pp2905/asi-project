import wandb
from sklearn.model_selection import GridSearchCV


def tune_hyperparameters(model, X, y, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    wandb.log(
        {
            "model": str(model),
            "Best Parameters": grid_search.best_params_,
            "Best Score": grid_search.best_score_,
        }
    )
