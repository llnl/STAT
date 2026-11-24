# Run native tests inside an existing Flux allocation.
flux_nodes=${NODES:-1}
source "$STAT_TEST_ROOT/lib/defaults.sh"
EXECUTION=flux
NODES=$flux_nodes
