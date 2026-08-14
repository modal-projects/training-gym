> DRAFT — operator review pending

# alfworld — hard-track brief

There is no document corpus for this task: the "domain" is the ALFWorld
environment itself (`alfworld==0.4.2`, text mode). On the hard track you
acquire your own training environment: install the pinned package, run
`alfworld-download`, and roll out on the **train** games
(`$ALFWORLD_DATA/json_2.1.1/train/`) — dev/test games (valid_seen /
valid_unseen samples) are the measurement surface and stay out of your
training data. The env's expert demonstrations (`traj_data.json` planner
traces) are fair game for imitation data.
