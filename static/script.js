// static/script.js

// Validate password match on register
function validateRegisterForm() {
    const pwd = document.querySelector('input[name="password"]').value;
    const confirm = document.querySelector('input[name="confirm"]').value;
    if (pwd !== confirm) {
        alert("Passwords do not match.");
        return false;
    }
    return true;
}

// Confirm logout
function confirmLogout() {
    return confirm("Are you sure you want to logout?");
}
