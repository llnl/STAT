"""Check Pause callbacks whose STAT session closes before or during execution.

Run the installed callback with fake GUI/session objects and mocked event
processing, without opening windows or controlling an application.
"""

from types import SimpleNamespace
import unittest
from unittest import mock

from STATGUI import STATGUI, STAT_OK


class CallbackReentrancyTests(unittest.TestCase):
    def test_pause_returns_if_a_queued_event_closes_the_session(self):
        """Simulate session cleanup during event processing.

        Check Pause returns successfully without polling the closed session.
        """
        stat = mock.Mock()
        stat.pause.return_value = STAT_OK
        gui = SimpleNamespace(STAT=stat, attached=True,
                              set_action_sensitivity=mock.Mock())

        def close_session():
            # Simulate a queued Detach clearing the session during event processing.
            gui.STAT = None

        with mock.patch('STATGUI.run_gtk_main_loop', side_effect=close_session):
            self.assertEqual(STATGUI.on_pause(gui), STAT_OK)
        self.assertIsNone(gui.STAT)
        stat.pause.assert_called_once_with()
        stat.receiveAck.assert_not_called()

    def test_pause_returns_if_the_session_is_already_closed(self):
        """Call Pause with no session but a previous attachment available.

        Check it requests detached controls and skips event processing.
        """
        gui = SimpleNamespace(STAT=None, attached=False, reattach=True,
                              set_action_sensitivity=mock.Mock())

        with mock.patch('STATGUI.run_gtk_main_loop') as dispatch:
            self.assertEqual(STATGUI.on_pause(gui), STAT_OK)
        # Keep ReAttach available and do not enter acknowledgement polling.
        gui.set_action_sensitivity.assert_called_once_with('detached')
        dispatch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
