//! Native `xn` command. Handles project scaffolding and environment checks itself,
//! and delegates training / generation to the Python engine via `uv run`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const CONFIG_FILE: &str = "xn.json";
const DEFAULT_CONFIG: &str = include_str!("../../templates/xn.json");
const SAMPLE_TEXT: &str = include_str!("../../templates/sample.txt");
const MIN_PYTHON: (u32, u32) = (3, 11);

const HELP: &str = "XN - Eazy make a LLM.

Usage: xn <command> [args]

Commands:
  init [dir]   create a new XN project (xn.json, data/sample.txt)
  doctor       check python, uv, the engine and the project
  data         fetch / list training data   (xn data --help)
  train        train a model from scratch   (xn train --help)
  generate     generate text from a model   (xn generate --help)

Environment:
  XN_ENGINE    path to the Python engine (directory containing pyproject.toml)
  XN_UV        uv executable to use (default: uv on PATH)";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("init") => init(Path::new(args.get(1).map_or(".", String::as_str))),
        Some("doctor") => Ok(doctor()),
        Some("train" | "generate" | "data") => run_engine(&args),
        Some("-h" | "--help" | "help") | None => {
            println!("{HELP}");
            Ok(ExitCode::SUCCESS)
        }
        Some("-V" | "--version") => {
            println!("xn {}", env!("CARGO_PKG_VERSION"));
            Ok(ExitCode::SUCCESS)
        }
        Some(other) => Err(format!("unknown command: {other}\n\n{HELP}")),
    };
    result.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        ExitCode::FAILURE
    })
}

fn init(root: &Path) -> Result<ExitCode, String> {
    let config = root.join(CONFIG_FILE);
    if config.exists() {
        return Err(format!("{} already exists", config.display()));
    }
    let write = |path: PathBuf, body: &str| {
        fs::write(&path, body).map_err(|e| format!("{}: {e}", path.display()))
    };
    fs::create_dir_all(root.join("data")).map_err(|e| e.to_string())?;
    write(root.join("data").join("sample.txt"), SAMPLE_TEXT)?;
    write(config, DEFAULT_CONFIG)?;
    write(root.join(".gitignore"), "runs/\n")?;
    let abs = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    println!("initialized XN project in {}", abs.display());
    println!("next: xn train  ->  xn generate --prompt \"...\"");
    Ok(ExitCode::SUCCESS)
}

/// Find an executable on PATH.
fn which(name: &str) -> Option<PathBuf> {
    env::split_paths(&env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

fn uv() -> Option<PathBuf> {
    env::var_os("XN_UV")
        .map(PathBuf::from)
        .or_else(|| which("uv"))
}

/// The Python engine: $XN_ENGINE, else the repository this binary was built from.
fn engine_dir() -> PathBuf {
    env::var_os("XN_ENGINE")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."))
}

/// Run a command and return its trimmed stdout+stderr (python 2 prints --version to stderr).
fn output(cmd: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    Some(text.trim().to_string())
}

fn engine_command(uv: &Path, args: &[String]) -> Command {
    let mut cmd = Command::new(uv);
    // stream progress immediately, even when output is redirected to a file
    cmd.env("PYTHONUNBUFFERED", "1")
        .args(["run", "--quiet", "--project"])
        .arg(engine_dir())
        .args(["python", "-m", "xn"])
        .args(args);
    cmd
}

fn run_engine(args: &[String]) -> Result<ExitCode, String> {
    let uv = uv().ok_or("uv not found; run `xn doctor` for install instructions")?;
    let status = engine_command(&uv, args)
        .status()
        .map_err(|e| format!("failed to run {}: {e}", uv.display()))?;
    Ok(ExitCode::from(status.code().unwrap_or(1) as u8))
}

fn report(ok: bool, name: &str, msg: &str) -> bool {
    println!("{} {name:8} {msg}", if ok { "✓" } else { "✗" });
    ok
}

fn check_python() -> bool {
    let Some(py) = which("python3").or_else(|| which("python")) else {
        return report(
            false,
            "python",
            "not found on PATH (uv can provide one: uv python install)",
        );
    };
    let version = output(&py, &["--version"]).unwrap_or_default();
    let parsed: Vec<u32> = version
        .trim_start_matches("Python ")
        .split('.')
        .filter_map(|n| n.parse().ok())
        .collect();
    let ok = matches!(parsed[..], [major, minor, ..] if (major, minor) >= MIN_PYTHON);
    let note = if ok {
        String::new()
    } else {
        format!(" - need >= {}.{}", MIN_PYTHON.0, MIN_PYTHON.1)
    };
    report(ok, "python", &format!("{version} ({}){note}", py.display()))
}

fn doctor() -> ExitCode {
    let mut ok = check_python();
    let uv = uv();
    ok &= match &uv {
        Some(uv) => {
            let version = output(uv, &["--version"]).unwrap_or_default();
            report(true, "uv", &format!("{version} ({})", uv.display()))
        }
        None => report(
            false,
            "uv",
            "not found - install: curl -LsSf https://astral.sh/uv/install.sh | sh",
        ),
    };
    let engine = engine_dir();
    let engine_ok = engine.join("pyproject.toml").is_file();
    let shown = fs::canonicalize(&engine).unwrap_or(engine);
    ok &= report(
        engine_ok,
        "engine",
        &if engine_ok {
            shown.display().to_string()
        } else {
            format!("{} not found - set XN_ENGINE", shown.display())
        },
    );

    // dependency / device / project checks run inside the engine's environment
    if let (Some(uv), true) = (&uv, engine_ok) {
        let status = engine_command(uv, &["doctor".into()]).status();
        ok &= matches!(status, Ok(s) if s.success());
    } else {
        println!("- skipped engine checks (need uv and the engine)");
        // the summary below already counts the failures above
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        println!("problems found");
        ExitCode::FAILURE
    }
}
