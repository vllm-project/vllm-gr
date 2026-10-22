//! CPU test peer: echo multipart messages over both shared-memory directions.
use ce_transport::iceoryx::{FrameReceiver, FrameSender, RECEIVE_WAIT};
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: iceoryx-peer TX RX CAPACITY COUNT".into());
    }
    let capacity = args[3].parse().map_err(|e| format!("capacity: {e}"))?;
    let count: usize = args[4].parse().map_err(|e| format!("count: {e}"))?;
    let receiver = FrameReceiver::new(&args[1], capacity)?;
    let sender = FrameSender::new(&args[2], capacity)?;
    println!("READY");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    // Tests can leave the registered receiver paused to exercise full buffers.
    let _ = std::io::stdin().lock().lines().next();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut received = 0;
    while received < count {
        if Instant::now() >= deadline {
            return Err("peer receive timeout".into());
        }
        if let Some(frames) = receiver.receive_wait(RECEIVE_WAIT)? {
            sender.send(&frames, Duration::from_secs(2), || false)?;
            received += 1;
        }
    }
    // Keep publisher alive until CE confirms receipt; no last-sample teardown race.
    println!("DONE");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let _ = std::io::stdin().lock().lines().next();
    Ok(())
}
