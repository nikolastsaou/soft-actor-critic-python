# Usage Instructions

## Requirements

Before performing any other action, the user must install the libraries listed in the `requirements.txt` file. For example, the `.txt` file can be placed in the `Source_Code` folder, after which the following command can be executed in the terminal:

```bash
pip install -r requirements.txt
```

> **Note:** Python **3.10** is recommended.

---

## Training

To train a new model, the user should open the **Source_Code** folder in an editor of their choice.

The `main_sac.py` file can then be used to specify the name of the environment to be trained, provided that the environment is supported by the **Gymnasium** and **MuJoCo** libraries.

The project also includes three files that already implement the training process for the following environments:

* `InvertedPendulum-v4`
* `HalfCheetah-v4`
* `Hopper-v5`

To train a model, simply configure the desired environment in the corresponding code and run the program.

---

## Testing

Once the model has been successfully trained, the trained agent can be tested by setting the following variable to `True`:

```python
load_checkpoint = True
```

Then, execute the same code again. The saved models will be loaded from the path specified by `chkpt_dir`.

A **render window** will then be displayed, allowing the user to visualize the interaction between the trained agent and the environment.
