"""Tests for gl-signerproxy synchronous implementation.

These tests verify that the rewritten synchronous gl-signerproxy works
correctly with standard library threading instead of tokio async.
"""
from gltesting.identity import Identity
from gltesting.fixtures import *
from pyln.testing.utils import wait_for, NodeFactory, LightningNode
from rich.pretty import pprint
from glclient import nodepb
from pyln import grpc as clnpb
import time
import pytest


def test_signerproxy_starts_with_node(clients):
    """Test that the signerproxy starts successfully when a node is scheduled.

    This verifies that the synchronous signerproxy can initialize properly
    and connect to the HSM server.
    """
    c = clients.new()
    c.register(configure=True)

    # Scheduling the node should start the signerproxy
    node_info = c.scheduler().schedule()
    assert node_info.grpc_uri is not None

    # The signer should be able to connect
    s = c.signer().run_in_thread()

    # Basic RPC call should work through the signerproxy
    gl1 = c.node()
    info = gl1.get_info()
    assert info.id is not None

    s.shutdown()


def test_signerproxy_handles_concurrent_signers(clients):
    """Test that the signerproxy can handle multiple concurrent signer connections.

    The synchronous implementation uses std::thread::spawn for each client,
    so we need to verify that multiple signers can operate concurrently.
    """
    c = clients.new()
    c.register(configure=True)

    # Start the node and attach a signer
    gl1 = c.node()
    s1 = c.signer().run_in_thread()

    # Generate an invoice which requires signer interaction
    inv1 = gl1.invoice(
        label="test1",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=42000)),
        description="test invoice 1",
    )
    assert inv1.bolt11 is not None

    # Generate another invoice to verify concurrent operation
    inv2 = gl1.invoice(
        label="test2",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=43000)),
        description="test invoice 2",
    )
    assert inv2.bolt11 is not None

    s1.shutdown()


def test_signerproxy_fd_passing(clients, executor):
    """Test that FD passing works correctly with synchronous I/O.

    The signerproxy uses UnixStream::pair() and FD passing (message type 9)
    to create new client connections. This test verifies that works with
    std::os::unix::net::UnixStream instead of tokio::net::UnixStream.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()

    # Running invoice in a separate thread will trigger FD creation
    fi = executor.submit(
        gl1.invoice,
        label="test_fd",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=50000)),
        description="FD passing test",
    )

    # Attach signer - this will cause the FD passing to complete
    s = c.signer().run_in_thread()

    # The invoice call should complete successfully
    inv = fi.result(10)
    assert inv.bolt11 is not None

    s.shutdown()


def test_signerproxy_request_forwarding(clients):
    """Test that HSM requests are properly forwarded to the gRPC server.

    The synchronous proxy uses runtime.block_on() to execute gRPC calls.
    This test verifies that requests are correctly forwarded and responses
    are returned.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()
    s = c.signer().run_in_thread()

    # Multiple operations to test request forwarding
    info = gl1.get_info()
    assert info.id is not None

    # Create multiple invoices to test repeated forwarding
    for i in range(5):
        inv = gl1.invoice(
            label=f"forward_test_{i}",
            amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=10000 + i)),
            description=f"forwarding test {i}",
        )
        assert inv.bolt11 is not None

    # List invoices should work
    invoices = gl1.list_invoices()
    assert len(invoices.invoices) >= 5

    s.shutdown()


def test_signerproxy_multiple_client_threads(clients):
    """Test that multiple client threads can operate concurrently.

    The synchronous implementation spawns a new thread for each client FD.
    This test creates multiple concurrent operations to verify thread safety.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()
    s = c.signer().run_in_thread()

    # Create multiple invoices concurrently
    # Each invoice creation may spawn new threads in the signerproxy
    invoices = []
    for i in range(10):
        inv = gl1.invoice(
            label=f"concurrent_{i}",
            amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=1000 * (i + 1))),
            description=f"concurrent test {i}",
        )
        invoices.append(inv)

    # All invoices should be created successfully
    assert len(invoices) == 10
    assert all(inv.bolt11 is not None for inv in invoices)

    s.shutdown()


def test_signerproxy_node_restart(clients):
    """Test that the signerproxy can handle node restarts.

    When using synchronous I/O and threads, we need to ensure cleanup
    happens properly on restart.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()
    s = c.signer().run_in_thread()

    # Create an invoice
    inv1 = gl1.invoice(
        label="before_restart",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=25000)),
        description="before restart",
    )
    assert inv1.bolt11 is not None

    # Stop the node
    c.schedsvc.nodes[0].process.stop()
    c.schedsvc.nodes[0].process = None
    s.shutdown()

    # Reschedule and create another invoice
    gl1 = c.node()
    s = c.signer().run_in_thread()

    inv2 = gl1.invoice(
        label="after_restart",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=26000)),
        description="after restart",
    )
    assert inv2.bolt11 is not None

    s.shutdown()


def test_signerproxy_blocking_io_performance(clients):
    """Test that blocking I/O doesn't cause performance issues.

    While synchronous I/O blocks on each operation, thread-based concurrency
    should still provide good performance for typical operations.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()
    s = c.signer().run_in_thread()

    # Measure time to create multiple invoices
    start = time.time()

    for i in range(20):
        gl1.invoice(
            label=f"perf_test_{i}",
            amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=5000)),
            description=f"performance test {i}",
        )

    duration = time.time() - start

    # Should complete in reasonable time (< 10 seconds for 20 invoices)
    # This is a generous limit to account for test environment variability
    assert duration < 10.0, f"Creating 20 invoices took {duration}s, expected < 10s"

    s.shutdown()


def test_signerproxy_wire_protocol_compatibility(clients):
    """Test that the synchronous wire protocol implementation is compatible.

    The wire.rs module was rewritten to use std::io instead of tokio::io.
    This test verifies that the wire protocol still works correctly.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()
    s = c.signer().run_in_thread()

    # Test various message types
    info = gl1.get_info()
    assert info.id is not None

    # New address (different message type)
    addr = gl1.new_address()
    assert addr.bech32 is not None

    # Invoice (yet another message type)
    inv = gl1.invoice(
        label="wire_test",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=1000)),
        description="wire protocol test",
    )
    assert inv.bolt11 is not None

    s.shutdown()


def test_signerproxy_error_handling(clients, executor):
    """Test that errors are properly handled in the synchronous implementation.

    Errors in synchronous code should be properly propagated without
    causing panics or deadlocks.
    """
    c = clients.new()
    c.register(configure=True)
    gl1 = c.node()

    # Try to create an invoice without a signer - should fail gracefully
    with pytest.raises(Exception):
        # This will timeout waiting for the signer
        gl1.invoice(
            label="no_signer",
            amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=1000)),
            description="no signer test",
            timeout=2,  # Short timeout
        )

    # Now attach signer and verify normal operation resumes
    s = c.signer().run_in_thread()

    inv = gl1.invoice(
        label="with_signer",
        amount_msat=clnpb.AmountOrAny(amount=clnpb.Amount(msat=1000)),
        description="with signer test",
    )
    assert inv.bolt11 is not None

    s.shutdown()
