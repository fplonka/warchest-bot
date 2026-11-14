# warchest-bot

A self-play bot for the two-player board game [War Chest](https://www.alderac.com/war-chest/).

War Chest has hidden information: each player draws coins from a private bag. The bot searches over information sets. At every simulation it deals the opponent's hidden coins from a belief that it updates after each move it sees. At the root it uses Gumbel sequential halving, as in Gumbel MuZero. A 12-layer transformer over the 37 hexes and the unit pools gives the value and the policy. The net trains on its own self-play games, with no human data.

## Layout

- `warchest/`: the rules. The `rules.md` file says how each card works and where that was checked.
- `engine/`: the search, the opponent belief, and the bot protocol that the ladder speaks.
- `bot/`: the net, its CUDA and Metal kernels, the self-play farm, the `bot` binary and the `warchest` Python module.
- `train/`: the PyTorch trainer. It runs self-play on every GPU and packs each snapshot as a bot.
- `tools/`: scripts to rent and drive a training box, Elo ladders between bots, and the draft table.

## Build

```sh
cargo test --release --features bot/metal        # macOS
cargo test --release --features bot/gpu          # CUDA
cd bot && maturin develop --release --features python,gpu
```

## Train

```sh
python train/train.py out=runs/mine minutes=600
```

This needs CUDA. With one GPU, add `device=cuda:0`. The knobs are in `train/config.py`. `tools/box.sh` sets up a rented box, syncs the tree and builds it.

## Draft table

`tools/draft_table.py` gives the bot's value of every possible draft. The table is attached to the [release](../../releases) as `draft.npz`. `tools/draft.py` drafts by exact minimax over it.

The trained weights are not in this repository.

## License

MIT
