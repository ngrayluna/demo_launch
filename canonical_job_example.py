import random
import wandb


def run_training_run(epochs, lr):
      print(f"Training for {epochs} epochs with learning rate {lr}")

      settings = wandb.Settings(enable_job_creation=True)
      run = wandb.init(
            # Set the project where this run will be logged
            project="launch_demo",
            # Track hyperparameters and run metadata
            job_type="eval",
            settings=settings,
            entity='wandb',
            config={
            "learning_rate": lr,
            "epochs": epochs,
            })

      offset = random.random() / 5
      print(f"lr: {lr}")

      for epoch in range(2, epochs):
            # simulating a training run
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset
            loss = 2 ** -epoch + random.random() / epoch + offset
            print(f"epoch={epoch}, acc={acc}, loss={loss}")
            wandb.log({"acc": acc, "loss": loss})

      run.log_code()
      run.finish()

run_training_run(epochs=10, lr=0.01)
