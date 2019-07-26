"""Support for tasks, coroutines and the scheduler."""

__all__ = ['Task']

import inspect
import weakref

from asyncio import base_tasks
from asyncio import compat
from asyncio import coroutines
from asyncio import futures

include 'futures.pyx'
#include 'loop.pxd'

# Weak set containing all tasks alive.
cdef object _all_tasks = weakref.WeakSet()

# Dictionary containing tasks that are currently active in
# all running event loops.  {EventLoop: Task}
cdef dict _current_tasks = {}


cdef class Task(Future):
    """A coroutine wrapped in a Future."""

    cdef public object _coro
    cdef public object _fut_waiter
    cdef public int _must_cancel
    cdef object __weakref__

    # If False, don't log a message if the task is destroyed whereas its
    # status is still pending
    cdef public int _log_destroy_pending


    # An important invariant maintained while a Task not done:
    #
    # - Either _fut_waiter is None, and _step() is scheduled;
    # - or _fut_waiter is some Future, and _step() is *not* scheduled.
    #
    # The only transition from the latter to the former is through
    # _wakeup().  When _fut_waiter is not None, one of its callbacks
    # must be _wakeup().

    def __init__(self, object coro, *, object loop=None):
        assert coroutines.iscoroutine(coro), repr(coro)
        super().__init__(loop=loop)
        if self._source_traceback:
            del self._source_traceback[-1]
        self._log_destroy_pending = 1
        self._log_traceback = 0
        self._coro = coro
        self._fut_waiter = None
        self._must_cancel = 0
        self._loop.call_soon(self._step)
        _all_tasks.add(self)

    @classmethod
    def current_task(cls, loop=None):
        """Return the currently running task in an event loop or None.

        By default the current task for the current event loop is returned.

        None is returned when called not in the context of a Task.
        """
        if loop is None:
            loop = events.get_event_loop()
        return _current_tasks.get(loop)

    @classmethod
    def all_tasks(cls, loop=None):
        """Return a set of all tasks for an event loop.

        By default all tasks for the current event loop are returned.
        """
        if loop is None:
            loop = events.get_event_loop()
        return {t for t in _all_tasks if t._loop is loop}


    # On Python 3.3 or older, objects with a destructor that are part of a
    # reference cycle are never destroyed. That's not the case any more on
    # Python 3.4 thanks to the PEP 442.
    if compat.PY34:
        def __del__(self):
            if self._state == futures._PENDING and self._log_destroy_pending:
                context = {
                    'task': self,
                    'message': 'Task was destroyed but it is pending!',
                }
                if self._source_traceback:
                    context['source_traceback'] = self._source_traceback
                self._loop.call_exception_handler(context)
            Future.__del__(self)

    def _repr_info(self):
        return base_tasks._task_repr_info(self)

    def get_stack(self, *, limit=None):
        """Return the list of stack frames for this task's coroutine.

        If the coroutine is not done, this returns the stack where it is
        suspended.  If the coroutine has completed successfully or was
        cancelled, this returns an empty list.  If the coroutine was
        terminated by an exception, this returns the list of traceback
        frames.

        The frames are always ordered from oldest to newest.

        The optional limit gives the maximum number of frames to
        return; by default all available frames are returned.  Its
        meaning differs depending on whether a stack or a traceback is
        returned: the newest frames of a stack are returned, but the
        oldest frames of a traceback are returned.  (This matches the
        behavior of the traceback module.)

        For reasons beyond our control, only one stack frame is
        returned for a suspended coroutine.
        """
        return base_tasks._task_get_stack(self, limit)

    def print_stack(self, *, limit=None, file=None):
        """Print the stack or traceback for this task's coroutine.

        This produces output similar to that of the traceback module,
        for the frames retrieved by get_stack().  The limit argument
        is passed to get_stack().  The file argument is an I/O stream
        to which the output is written; by default output is written
        to sys.stderr.
        """
        return base_tasks._task_print_stack(self, limit, file)

    def cancel(self):
        """Request that this task cancel itself.

        This arranges for a CancelledError to be thrown into the
        wrapped coroutine on the next cycle through the event loop.
        The coroutine then has a chance to clean up or even deny
        the request using try/except/finally.

        Unlike Future.cancel, this does not guarantee that the
        task will be cancelled: the exception might be caught and
        acted upon, delaying cancellation of the task or preventing
        cancellation completely.  The task may also return a value or
        raise a different exception.

        Immediately after this method is called, Task.cancelled() will
        not return True (unless the task was already cancelled).  A
        task will be marked as cancelled when the wrapped coroutine
        terminates with a CancelledError exception (even if cancel()
        was not called).
        """
        self._log_traceback = 0
        if self.done():
            return False
        if self._fut_waiter is not None:
            if self._fut_waiter.cancel():
                # Leave self._fut_waiter; it may be a Task that
                # catches and ignores the cancellation so we may have
                # to cancel it again later.
                return True
        # It must be the case that self._step is already scheduled.
        self._must_cancel = 1
        return True

    cpdef _step(self, Exception exc = None):
        curr_task = _current_tasks.get(self._loop)
        try:
            self._step_orig(exc)
        finally:
            if curr_task is None:
                _current_tasks.pop(self._loop, None)
            else:
                _current_tasks[self._loop] = curr_task

    cpdef _step_orig(self, Exception exc = None):
        assert not self.done(), \
            '_step(): already done: {!r}, {!r}'.format(self, exc)
        if self._must_cancel:
            if not isinstance(exc, futures.CancelledError):
                exc = futures.CancelledError()
            self._must_cancel = 0
        coro = self._coro
        self._fut_waiter = None

        _current_tasks[self._loop] = self
        # Call either coro.throw(exc) or coro.send(None).
        try:
            if exc is None:
                # We use the `send` method directly, because coroutines
                # don't have `__iter__` and `__next__` methods.
                result = coro.send(None)
            else:
                result = coro.throw(exc)
        except StopIteration as exc:
            if self._must_cancel:
                # Task is cancelled right before coro stops.
                self._must_cancel = 0
                self.set_exception(futures.CancelledError())
            else:
                self.set_result(exc.value)
        except futures.CancelledError:
            super().cancel()  # I.e., Future.cancel(self).
        except Exception as exc:
            self.set_exception(exc)
        except BaseException as exc:
            self.set_exception(exc)
            raise
        else:
            blocking = getattr(result, '_asyncio_future_blocking', None)
            if blocking is not None:
                # Yielded Future must come from Future.__iter__().
                if result._loop is not self._loop:
                    self._loop.call_soon(
                        self._step,
                        RuntimeError(
                            'Task {!r} got Future {!r} attached to a '
                            'different loop'.format(self, result)))
                elif blocking:
                    if result is self:
                        self._loop.call_soon(
                            self._step,
                            RuntimeError(
                                'Task cannot await on itself: {!r}'.format(
                                    self)))
                    else:
                        result._asyncio_future_blocking = 0
                        result.add_done_callback(self._wakeup)
                        self._fut_waiter = result
                        if self._must_cancel:
                            if self._fut_waiter.cancel():
                                self._must_cancel = 0
                else:
                    self._loop.call_soon(
                        self._step,
                        RuntimeError(
                            'yield was used instead of yield from '
                            'in task {!r} with {!r}'.format(self, result)))
            elif result is None:
                # Bare yield relinquishes control for one event loop iteration.
                self._loop.call_soon(self._step)
            elif inspect.isgenerator(result):
                # Yielding a generator is just wrong.
                self._loop.call_soon(
                    self._step,
                    RuntimeError(
                        'yield was used instead of yield from for '
                        'generator in task {!r} with {}'.format(
                            self, result)))
            else:
                # Yielding something else is an error.
                self._loop.call_soon(
                    self._step,
                    RuntimeError(
                        'Task got bad yield: {!r}'.format(result)))
        finally:
            _current_tasks.pop(self._loop)
            self = None  # Needed to break cycles when an exception occurs.

    def _wakeup(self, future):
        try:
            future.result()
        except Exception as exc:
            # This may also be a cancellation.
            self._step(exc)
        else:
            # Don't pass the value of `future.result()` explicitly,
            # as `Future.__iter__` and `Future.__await__` don't need it.
            # If we call `_step(value, None)` instead of `_step()`,
            # Python eval loop would use `.send(value)` method call,
            # instead of `__next__()`, which is slower for futures
            # that return non-generator iterators from their `__iter__`.
            self._step()
        self = None  # Needed to break cycles when an exception occurs.


# _PyTask = Task
#
#
# try:
#     import _asyncio
# except ImportError:
#     pass
# else:
#     # _CTask is needed for tests.
#     Task = _CTask = _asyncio.Task
