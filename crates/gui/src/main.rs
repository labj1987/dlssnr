mod binaries;
mod shm;
mod ui;

use gtk4::prelude::*;

fn main() {
    // Set program name before GTK init. On Wayland the app_id GNOME sees is the
    // GApplication ID, not prgname; on X11 it's prgname. Setting both prgname and
    // StartupWMClass (in the .desktop file) to the application ID makes the running
    // window match the desktop file on either backend.
    glib::set_prgname(Some("io.github.labj1987.Dlssnr"));
    glib::set_application_name("dlssnr");

    let app = libadwaita::Application::builder()
        .application_id("io.github.labj1987.Dlssnr")
        .flags(gio::ApplicationFlags::FLAGS_NONE)
        .build();

    app.connect_activate(|app| {
        if let Some(window) = app.windows().first() {
            window.present();
            return;
        }
        ui::build_ui(app);
    });

    std::process::exit(app.run().get() as i32);
}
