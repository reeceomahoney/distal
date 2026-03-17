"""Trim a wandb run by re-logging only data up to a given step into a new run."""

import sys
import time

import wandb

ENTITY = "reeceomahoney"
PROJECT = "distal"
RUN_ID = "h0ve6zjj"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <cutoff_step>")
        sys.exit(1)

    cutoff_step = int(sys.argv[1])

    api = wandb.Api()
    old_run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    print(f"Fetching history from run '{old_run.name}' up to step {cutoff_step}...")

    rows = []
    for row in old_run.scan_history():
        if row["_step"] > cutoff_step:
            break
        rows.append(row)

    print(f"Collected {len(rows)} rows. Re-logging to new run...")

    new_run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        config=old_run.config,
        name=f"{old_run.name}-trimmed-{cutoff_step}",
    )

    # Calculate the exact time shift needed to align the old timeline with the new run
    if rows and "_timestamp" in rows[0]:
        # Approximate when the old run started
        old_start_time = rows[0]["_timestamp"] - rows[0].get("_runtime", 0)
        # Calculate how far forward we need to shift the timestamps
        time_shift = time.time() - old_start_time
    else:
        time_shift = 0

    for row in rows:
        step = row.pop("_step")

        # Shift the absolute timestamp forward to the new run's timeline
        if "_timestamp" in row:
            row["_timestamp"] += time_shift

        new_run.log(row, step=step)

    wandb.finish()
    print("Done. New trimmed run created.")


if __name__ == "__main__":
    main()
