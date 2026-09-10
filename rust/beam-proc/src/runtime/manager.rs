use crossbeam_channel::{Receiver, SendError, Sender, TryRecvError, bounded};
use std::thread::{self, JoinHandle};

const DEFAULT_MAX_MESSAGES_PER_TICK: usize = 1024;

pub trait ManagerMessage {
    fn shutdown() -> Self;
    fn is_shutdown(&self) -> bool;
}

pub enum ManagerEvent<M> {
    Message(M),
    Idle,
}

enum ManagerMode {
    Blocking,
    NonBlocking { max_messages_per_tick: usize },
}

pub struct Manager<M>
where
    M: ManagerMessage + Send + 'static,
{
    tx: Sender<M>,
    rx: Option<Receiver<M>>,
    handler: Option<Box<dyn FnMut(ManagerEvent<M>) + Send + 'static>>,
    mode: ManagerMode,
    handle: Option<JoinHandle<()>>,
}

impl<M> Manager<M>
where
    M: ManagerMessage + Send + 'static,
{
    pub fn init_with_capacity<F>(handler: F, capacity: usize) -> Self
    where
        F: FnMut(M) + Send + 'static,
    {
        let mut handler = handler;
        let (tx, rx) = bounded(capacity.max(1));

        Self {
            tx,
            rx: Some(rx),
            handler: Some(Box::new(move |event| {
                if let ManagerEvent::Message(msg) = event {
                    handler(msg);
                }
            })),
            mode: ManagerMode::Blocking,
            handle: None,
        }
    }

    pub fn init_nonblocking_with_capacity<F>(handler: F, capacity: usize) -> Self
    where
        F: FnMut(ManagerEvent<M>) + Send + 'static,
    {
        let (tx, rx) = bounded(capacity.max(1));

        Self {
            tx,
            rx: Some(rx),
            handler: Some(Box::new(handler)),
            mode: ManagerMode::NonBlocking {
                max_messages_per_tick: DEFAULT_MAX_MESSAGES_PER_TICK,
            },
            handle: None,
        }
    }

    pub fn message_queue(&self) -> Sender<M> {
        self.tx.clone()
    }

    pub fn run(&mut self) -> anyhow::Result<()> {
        let rx = self.rx.take().expect("manager already running");
        let mut handler = self.handler.take().expect("manager handler already taken");
        let mode = std::mem::replace(&mut self.mode, ManagerMode::Blocking);
        let (ready_tx, ready_rx) = bounded(1);

        let handle = thread::Builder::new()
            .spawn(move || {
                let _ = ready_tx.send(());
                match mode {
                    ManagerMode::Blocking => run_blocking(rx, &mut handler),
                    ManagerMode::NonBlocking {
                        max_messages_per_tick,
                    } => run_nonblocking(rx, &mut handler, max_messages_per_tick),
                }
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn manager thread: {err}"))?;

        self.handle = Some(handle);
        ready_rx
            .recv()
            .map_err(|err| anyhow::anyhow!("manager thread exited before READY: {err}"))
    }

    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_some_and(JoinHandle::is_finished)
    }

    pub fn shutdown(&self) -> Result<(), SendError<M>> {
        self.tx.send(M::shutdown())
    }

    pub fn join(&mut self) -> thread::Result<()> {
        if let Some(handle) = self.handle.take() {
            return handle.join();
        }

        Ok(())
    }
}

fn run_blocking<M>(rx: Receiver<M>, handler: &mut Box<dyn FnMut(ManagerEvent<M>) + Send + 'static>)
where
    M: ManagerMessage + Send + 'static,
{
    loop {
        match rx.recv() {
            Ok(msg) if msg.is_shutdown() => break,
            Ok(msg) => handler(ManagerEvent::Message(msg)),
            Err(_) => break,
        }
    }
}

fn run_nonblocking<M>(
    rx: Receiver<M>,
    handler: &mut Box<dyn FnMut(ManagerEvent<M>) + Send + 'static>,
    max_messages_per_tick: usize,
) where
    M: ManagerMessage + Send + 'static,
{
    let max_messages_per_tick = max_messages_per_tick.max(1);

    loop {
        for _ in 0..max_messages_per_tick {
            match rx.try_recv() {
                Ok(msg) if msg.is_shutdown() => return,
                Ok(msg) => handler(ManagerEvent::Message(msg)),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }

        handler(ManagerEvent::Idle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    enum TestMessage {
        Panic,
        Shutdown,
    }

    impl ManagerMessage for TestMessage {
        fn shutdown() -> Self {
            Self::Shutdown
        }

        fn is_shutdown(&self) -> bool {
            matches!(self, Self::Shutdown)
        }
    }

    #[test]
    fn manager_reports_worker_panic_after_ready() {
        let mut manager = Manager::init_with_capacity(
            |message| match message {
                TestMessage::Panic => panic!("test worker panic"),
                TestMessage::Shutdown => {}
            },
            64,
        );
        manager.run().unwrap();
        manager.message_queue().send(TestMessage::Panic).unwrap();

        let deadline = Instant::now() + Duration::from_secs(1);
        while !manager.is_finished() && Instant::now() < deadline {
            thread::yield_now();
        }

        assert!(manager.is_finished());
        assert!(manager.join().is_err());
    }

    #[test]
    fn manager_queue_capacity_is_bounded() {
        let manager = Manager::init_with_capacity(|_message: TestMessage| {}, 2);
        let queue = manager.message_queue();

        queue.try_send(TestMessage::Panic).unwrap();
        queue.try_send(TestMessage::Panic).unwrap();
        assert!(queue.try_send(TestMessage::Panic).is_err());
        assert_eq!(queue.capacity(), Some(2));
    }
}
