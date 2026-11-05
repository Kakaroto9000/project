// News Trader's native window. On start it launches the Python engine + API
// (python -m newsmodel.live.server, see src/newsmodel/live/server.py) unless one is
// already running on 127.0.0.1:8765, and stops it again when the window closes.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::os::windows::process::CommandExt;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, RunEvent};

// This computer's Python and the repo; override with NEWSTRADER_PYTHONW / NEWSTRADER_ROOT
const PYTHONW: &str = r"C:\Users\oleks\AppData\Local\Programs\Python\Python313\pythonw.exe";
const ROOT: &str = r"C:\project";
const SERVER: &str = "127.0.0.1:8765";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The server this window started (None if one was already running, e.g. a second window).
struct Server(Mutex<Option<Child>>);

fn server_running() -> bool {
    TcpStream::connect_timeout(&SERVER.parse().unwrap(), Duration::from_millis(300)).is_ok()
}

fn start_server() -> Option<Child> {
    if server_running() {
        return None;
    }
    let python = std::env::var("NEWSTRADER_PYTHONW").unwrap_or_else(|_| PYTHONW.into());
    let root = std::env::var("NEWSTRADER_ROOT").unwrap_or_else(|_| ROOT.into());
    Command::new(python)
        .args(["-m", "newsmodel.live.server"])
        .current_dir(root)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .ok()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Server(Mutex::new(start_server())))
        .build(tauri::generate_context!())
        .expect("error while starting News Trader")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(mut child) = app.state::<Server>().0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
