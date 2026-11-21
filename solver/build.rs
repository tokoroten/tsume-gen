//! Embeds the commit the solver is built from as `TSUME_SOLVER_BUILD`
//! (`tsume_solver::record_clock::SOLVER_BUILD`): the catalog records'
//! `producer.solverBuild` and the default `--record-solver-id`.
//!
//! The value is the first 7 hex digits of `HEAD`, with `+dirty` when a file of
//! this crate (`solver/`, tracked or not, ignored files aside)
//! differs from the commit; `unknown` when git cannot be run here (a source
//! tarball).  `TSUME_SOLVER_BUILD` set at build time overrides it (a bundle
//! built from a tarball of a known commit).
//!
//! The script runs again when `HEAD` moves (its ref, `packed-refs`) or a file
//! of the crate other than `target*` changes, so `+dirty` follows the work
//! tree.  It changes nothing else the solver writes.

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

/// Reruns the script when the git file `name` (`HEAD`, a ref) changes.
fn rerun_on_git_path(dir: &Path, name: &str) {
    if let Some(path) = git(dir, &["rev-parse", "--git-path", name]) {
        let path = dir.join(path);
        if path.exists() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn detect(dir: &Path) -> String {
    // Every file of the crate but the build outputs: `+dirty` must follow an
    // edit.  (Naming any path turns Cargo's default, "any file of the
    // package", off; `target` changes on every build.)
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut names: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                !name.starts_with("target") && !name.starts_with('.')
            })
            .map(|entry| entry.path())
            .collect();
        names.sort();
        for path in names {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
    // Only the repository that tracks this crate: a `git archive` tree
    // unpacked inside another repository (a remote machine's work directory) must
    // not take that repository's commit.
    if git(dir, &["ls-files", "--error-unmatch", "Cargo.toml"]).is_none() {
        return "unknown".to_string();
    }
    let Some(head) = git(dir, &["rev-parse", "HEAD"]) else {
        return "unknown".to_string();
    };
    if head.len() < 7 || !head.bytes().all(|b| b.is_ascii_hexdigit()) {
        return "unknown".to_string();
    }
    rerun_on_git_path(dir, "HEAD");
    rerun_on_git_path(dir, "packed-refs");
    if let Some(reference) = git(dir, &["symbolic-ref", "-q", "HEAD"]) {
        rerun_on_git_path(dir, &reference);
    }
    let commit = head[..7].to_ascii_lowercase();
    match git(dir, &["status", "--porcelain", "--", "."]) {
        Some(status) if status.is_empty() => commit,
        Some(_) => format!("{commit}+dirty"),
        // A commit whose work tree cannot be read is not known to be clean.
        None => format!("{commit}+dirty"),
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TSUME_SOLVER_BUILD");
    let dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by Cargo"));
    let build = match std::env::var("TSUME_SOLVER_BUILD") {
        Ok(value) if !value.trim().is_empty() => {
            let value = value.trim().to_string();
            let (commit, rest) = value.split_at(value.len().min(7));
            let commit_ok = commit.len() == 7
                && commit
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
            assert!(
                (commit_ok && (rest.is_empty() || rest == "+dirty")) || value == "unknown",
                "TSUME_SOLVER_BUILD must be <7 lowercase hex digits>[+dirty] or unknown, got {value}"
            );
            value
        }
        _ => detect(&dir),
    };
    println!("cargo:rustc-env=TSUME_SOLVER_BUILD={build}");
}
