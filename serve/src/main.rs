//! `cargo web` builds the four WebAssembly modules and serves the demo site
//! on http://localhost:8000. `cargo web --out DIR` writes the site to DIR
//! instead, for any static host; the Pages workflow does that.
//!
//! A development tool, so it uses std. The demos do not.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::Arc;
use std::{env, fs, thread};

const STEPS: [&str; 4] = ["sine", "gesture", "keywords", "tinystories"];

fn main() -> ExitCode {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("repository root");
    let (mut out, mut port) = (None, 8000);
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match (arg.as_str(), args.next()) {
            ("--out", Some(dir)) => out = Some(PathBuf::from(dir)),
            ("--port", Some(p)) if p.parse::<u16>().is_ok() => port = p.parse().unwrap(),
            _ => {
                eprintln!("usage: cargo web [--port N] [--out DIR]");
                return ExitCode::FAILURE;
            }
        }
    }
    if !build(&root.join("web")) {
        return ExitCode::FAILURE;
    }
    match out {
        Some(dir) => export(&site(root), &dir),
        None => serve(Arc::new(root.to_path_buf()), port),
    }
}

fn build(web: &Path) -> bool {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    println!("building the WebAssembly modules");
    // An explicit target directory, so a global CARGO_TARGET_DIR cannot move the modules.
    let status = Command::new(&cargo)
        .args(["build", "--release", "--target-dir"])
        .arg(web.join("target"))
        .current_dir(web)
        .status();
    match status {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("cannot run {}: {e}", cargo.to_string_lossy());
            false
        }
    }
}

/// Every URL of the site with its file, for `--out`.
fn site(root: &Path) -> BTreeMap<String, PathBuf> {
    let mut urls = Vec::new();
    add_dir(&root.join("web/site"), "", &mut urls);
    urls.extend(STEPS.map(|step| format!("/{step}.wasm")));
    urls.extend(["/enroll.i16", "/stream.i16"].map(String::from));
    urls.into_iter().filter_map(|url| Some((url.clone(), file(root, &url)?))).collect()
}

fn add_dir(dir: &Path, prefix: &str, urls: &mut Vec<String>) {
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let url = format!("{prefix}/{}", entry.file_name().to_string_lossy());
        if entry.path().is_dir() {
            add_dir(&entry.path(), &url, urls);
        } else {
            urls.push(url);
        }
    }
}

fn export(files: &BTreeMap<String, PathBuf>, dir: &Path) -> ExitCode {
    for (url, file) in files {
        let target = dir.join(url.trim_start_matches('/'));
        let copied = target.parent().map_or(Ok(()), fs::create_dir_all).and_then(|()| fs::copy(file, &target));
        if let Err(e) = copied {
            eprintln!("{}: {e}", file.display());
            return ExitCode::FAILURE;
        }
    }
    println!("site written to {}", dir.display());
    ExitCode::SUCCESS
}

fn serve(root: Arc<PathBuf>, port: u16) -> ExitCode {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("port {port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("http://localhost:{port}/   (Ctrl-C stops)");
    for stream in listener.incoming().flatten() {
        let root = Arc::clone(&root);
        thread::spawn(move || respond(stream, &root));
    }
    ExitCode::SUCCESS
}

fn respond(mut stream: TcpStream, root: &Path) {
    let mut path = {
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() {
            return;
        }
        let path = line.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("/").to_owned();
        // Drain the headers; nothing in them matters here.
        while reader.read_line(&mut line).is_ok_and(|n| n > 2) {
            line.clear();
        }
        path
    };
    if path.ends_with('/') {
        path.push_str("index.html");
    }
    // Resolved on every request, so new and edited files show up on reload.
    let body = file(root, &path).and_then(|f| fs::read(f).ok());
    let (status, kind) = match body {
        Some(_) => ("200 OK", content_type(&path)),
        None => ("404 Not Found", "text/plain; charset=utf-8"),
    };
    let body = body.unwrap_or_else(|| format!("{path} not found\n").into_bytes());
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).and_then(|()| stream.write_all(&body));
}

/// The file behind a URL path: a module, a sample recording or a site file.
fn file(root: &Path, url: &str) -> Option<PathBuf> {
    let name = url.strip_prefix('/')?;
    if name.split('/').any(|part| part.is_empty() || part.starts_with('.')) {
        return None; // no "..", no hidden files
    }
    if let Some(step) = name.strip_suffix(".wasm").filter(|s| STEPS.contains(s)) {
        return Some(root.join(format!("web/target/wasm32-unknown-unknown/release/{step}_web.wasm")));
    }
    if let Some(clip) = name.strip_suffix(".i16").filter(|c| ["enroll", "stream"].contains(c)) {
        return Some(root.join(format!("3-keywords/testdata/{clip}.i16")));
    }
    Some(root.join("web/site").join(name))
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        _ => "application/octet-stream",
    }
}
