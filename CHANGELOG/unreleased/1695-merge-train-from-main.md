### CI

- **Merge-train now runs only code from `main` ([#1695](https://github.com/robotrocketscience/aelfrice/issues/1695)).**
  - **Before:** the train triggered on `pull_request`, so its workflow file and the gate scripts it runs came from the pull request being judged, with a `contents: write` token. A pull request could change the gate that decided whether it merged.
  - **Trigger:** the train now triggers on `pull_request_target`. GitHub takes that workflow file from the default branch, the train checks out `main`, and it fetches the pull request's head only as git data, by an explicit refspec.
  - **Forks:** for a pull request from a fork, the workflow starts, but the merge job is skipped.
  - **Concurrency:** it moved from the workflow to the merge job. A run that skips the job, such as a push to an unlabelled pull request, no longer joins the queue, so it can't displace a labelled run that's waiting for its slot.
  - **Conflicted pull requests:** a labelled pull request with a merge conflict now gets the not-fast-forward refusal comment. Before, `pull_request` didn't run for it.
  - **Stacked pull requests:** they still get the refusal from [#1424](https://github.com/robotrocketscience/aelfrice/issues/1424).
