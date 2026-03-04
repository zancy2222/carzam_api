<?php
// Set timezone to Asia/Manila (GMT+8)
date_default_timezone_set('Asia/Manila');
// Include database connection
require_once 'partials/db_conn.php';
session_start();

// ---------------------------------------------------------------------
// 1. AUTHENTICATION
// ---------------------------------------------------------------------
if (!isset($_SESSION['logged_in']) || $_SESSION['logged_in'] !== true) {
    header("Location: login.php");
    exit;
}
$user_id   = $_SESSION['user_id'] ?? null;
$user_name = $_SESSION['user_name'] ?? 'User';
$destinations = ['Inside Zambales', 'Outside Zambales', 'Baguio'];

// ---------------------------------------------------------------------
// 2. BOOKING LOGIC (ENHANCED with trip purpose & seaters)
// ---------------------------------------------------------------------
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['book_car'])) {
    $car_id      = filter_var($_POST['car_id'], FILTER_VALIDATE_INT);
    $duration    = $_POST['duration'] ?? '12hrs';
    $destination = $conn->real_escape_string($_POST['destination']);
    $address     = $conn->real_escape_string($_POST['address']);
    $return_date = $conn->real_escape_string($_POST['return_date']);

    // NEW FIELDS
    $trip_purpose = $conn->real_escape_string($_POST['trip_purpose'] ?? '');
    $number_of_seaters = filter_var($_POST['number_of_seaters'], FILTER_VALIDATE_INT);
    $terms_accepted = isset($_POST['terms_accepted']) ? 1 : 0;

    // Validate terms acceptance
    if (!$terms_accepted) {
        $_SESSION['booking_error'] = "You must accept the terms and conditions to proceed.";
        header("Location: " . $_SERVER['PHP_SELF']);
        exit;
    }

    // FIX 1: Use the booking_date the user selected, not server NOW.
    //         PHP was using date('Y-m-d H:i:s') which made the duration
    //         start from "right now" instead of the chosen booking time,
    //         causing a longer (and more expensive) duration than what JS showed.
    $raw_booking_date = $conn->real_escape_string($_POST['booking_date'] ?? '');
    $booking_date     = $raw_booking_date
                        ? date('Y-m-d H:i:s', strtotime($raw_booking_date))
                        : date('Y-m-d H:i:s');

    $booking_datetime = new DateTime($booking_date);
    $return_datetime  = new DateTime($return_date);

    if ($return_datetime <= $booking_datetime) {
        $_SESSION['booking_error'] = "Return date must be after booking date.";
        header("Location: " . $_SERVER['PHP_SELF']);
        exit;
    }

    $interval = $booking_datetime->diff($return_datetime);
    // FIX 2: Include minutes so ceil() matches the JS calculation exactly.
    //         Old code dropped $interval->i, making PHP round up one extra
    //         multiplier when minutes pushed the total just past a 12/24-hr boundary.
    $hours    = $interval->days * 24 + $interval->h + ($interval->i / 60);

    // Rate calculation logic (unchanged)
    if ($destination == 'Baguio') {
        $rate_field = $duration == '12hrs' ? 'rate_baguio_12hrs' : 'rate_baguio_24hrs';
    } elseif ($destination == 'Inside Zambales') {
        $rate_field = $duration == '12hrs' ? 'rate_inside_zambales_12hrs' : 'rate_inside_zambales_24hrs';
    } else {
        $rate_field = $duration == '12hrs' ? 'rate_outside_zambales_12hrs' : 'rate_outside_zambales_24hrs';
    }

    $driver_field = $duration == '12hrs' ? 'driver_price_12hrs' : 'driver_price_24hrs';

    $query = "SELECT $rate_field AS base_rate, with_driver, COALESCE($driver_field, 0) AS driver_price FROM car_rates WHERE car_id = $car_id";
    $rate_result = $conn->query($query);
    $rate_data = $rate_result && $rate_result->num_rows > 0 ? $rate_result->fetch_assoc() : null;

    if (!$rate_data || $rate_data['base_rate'] === null) {
        $avg_query = "SELECT AVG($rate_field) AS avg_rate FROM car_rates WHERE $rate_field IS NOT NULL";
        $avg_result = $conn->query($avg_query);
        $base_rate   = $avg_result->fetch_assoc()['avg_rate'] ?? 0;
        $with_driver = false;
        $driver_price = 0;
    } else {
        $base_rate    = $rate_data['base_rate'];
        $with_driver  = $rate_data['with_driver'];
        $driver_price = $rate_data['driver_price'];
    }

    $selected_duration_hours = $duration == '12hrs' ? 12 : 24;
    $multiplier = ceil($hours / $selected_duration_hours);
    $total_price = ($base_rate * $multiplier) + ($with_driver ? $driver_price * $multiplier : 0);

    if ($car_id && $total_price && $user_id && $address && $destination && $return_date && $number_of_seaters) {
        $avail_query = "SELECT status FROM car_availability WHERE car_id = $car_id";
        $avail_result = $conn->query($avail_query);

        if ($avail_result && $avail_result->num_rows > 0 && $avail_result->fetch_assoc()['status'] === 'available') {
            // Updated INSERT with new fields
            $booking_query = "INSERT INTO bookings 
                (user_id, car_id, booking_status, payment_status, booking_date, return_date, 
                 total_price, duration, destination, address, trip_purpose, number_of_seaters, created_at)
                VALUES 
                ($user_id, $car_id, 'pending', 'unpaid', '$booking_date', '$return_date', 
                 $total_price, '$duration', '$destination', '$address', '$trip_purpose', $number_of_seaters, CURRENT_TIMESTAMP)";

            if ($conn->query($booking_query)) {
                $stakeholder_query = "SELECT stakeholder_id FROM car_inventory WHERE id = $car_id";
                $stakeholder_result = $conn->query($stakeholder_query);
                $stakeholder_id = $stakeholder_result && $stakeholder_result->num_rows > 0 ? $stakeholder_result->fetch_assoc()['stakeholder_id'] : null;

                if ($stakeholder_id) {
                    $history_query = "INSERT INTO booking_history 
                        (car_id, user_id, stakeholder_id, start_date, end_date, booking_status, payment_status, total_amount)
                        VALUES 
                        ($car_id, $user_id, $stakeholder_id, '$booking_date', '$return_date', 'pending', 'unpaid', $total_price)";
                    $conn->query($history_query);
                }

                $update_avail_query = "UPDATE car_availability SET status = 'dispatched', updated_at = CURRENT_TIMESTAMP WHERE car_id = $car_id";
                $conn->query($update_avail_query);

                $_SESSION['booking_success'] = "Booking successfully confirmed! Awaiting stakeholder approval.";
            } else {
                $_SESSION['booking_error'] = "Failed to create booking. Please try again.";
            }
        } else {
            $_SESSION['booking_error'] = "Selected car is not available for booking.";
        }
    } else {
        $_SESSION['booking_error'] = "Invalid booking details. Please fill all required fields.";
    }

    header("Location: " . $_SERVER['PHP_SELF']);
    exit;
}

// ---------------------------------------------------------------------
// 3. DEFAULT CARS + PAGINATION
// FIX: Added ALL individual rate columns so booking modal hidden inputs
//      can read the real values instead of falling back to hardcoded defaults.
// ---------------------------------------------------------------------
$results_per_page = 6;
$current_page     = max(1, (int)($_GET['page'] ?? 1));

$query = "SELECT
            ci.*,
            sa.shop_name,
            sa.location,
            COALESCE(cr.rate_inside_zambales_12hrs, 3000) AS rate,
            cr.with_driver,
            COALESCE(cr.driver_price_12hrs, 500)          AS driver_price,
            COALESCE(cr.rate_inside_zambales_12hrs,  3000) AS rate_inside_zambales_12hrs,
            COALESCE(cr.rate_inside_zambales_24hrs,  5000) AS rate_inside_zambales_24hrs,
            COALESCE(cr.rate_outside_zambales_12hrs, 3500) AS rate_outside_zambales_12hrs,
            COALESCE(cr.rate_outside_zambales_24hrs, 6000) AS rate_outside_zambales_24hrs,
            COALESCE(cr.rate_baguio_12hrs,           4000) AS rate_baguio_12hrs,
            COALESCE(cr.rate_baguio_24hrs,           7000) AS rate_baguio_24hrs,
            COALESCE(cr.driver_price_12hrs,           500) AS driver_price_12hrs,
            COALESCE(cr.driver_price_24hrs,           800) AS driver_price_24hrs,
            cat.trip_purpose,
            cat.seating_capacity,
            cat.terrain_compatibility,
            cat.ground_clearance,
            cat.cargo_space,
            cat.body_size,
            cat.compact,
            (SELECT COUNT(*) FROM booking_history bh
             WHERE bh.car_id = ci.id
               AND bh.booking_status IN ('confirmed','completed')) AS booking_count
          FROM car_inventory ci
          JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
          LEFT JOIN car_rates cr ON ci.id = cr.car_id
          LEFT JOIN car_attributes cat ON ci.id = cat.car_id
          JOIN car_availability ca ON ci.id = ca.car_id
          WHERE ca.status = 'available'
          ORDER BY booking_count DESC
          LIMIT " . (($current_page - 1) * $results_per_page) . ", $results_per_page";
$result = $conn->query($query);

$count_query   = "SELECT COUNT(*) as total FROM car_inventory ci JOIN car_availability ca ON ci.id = ca.car_id WHERE ca.status = 'available'";
$count_result  = $conn->query($count_query);
$total_records = $count_result->fetch_assoc()['total'];
$total_pages   = ceil($total_records / $results_per_page);

// ---------------------------------------------------------------------
// 4. FLAG: are we showing RECOMMENDED cars? (used in HTML)
// ---------------------------------------------------------------------
$is_recommendation = false;   // default = false (normal pagination view)
?>
<?php
// Enhanced Search with JSON + Reviews + Car Attributes (FIXED VERSION)
$search_query = '';
$search_results = null;
$search_performed = false;
$total_search_results = 0;

if (isset($_GET['search']) && trim($_GET['search']) !== '') {
    $search_query = trim($_GET['search']);
    $search_performed = true;

    // Sanitize input
    $like_term = "%" . $conn->real_escape_string($search_query) . "%";
    $json_term = $conn->real_escape_string($search_query);

    // Normalize search term for better matching
    $lower_search = strtolower($search_query);

    // Build dynamic WHERE conditions with smart keyword mapping
    $where_conditions = [];

    // 1. Basic text fields
    $where_conditions[] = "ci.car_name LIKE '$like_term'";
    $where_conditions[] = "sa.shop_name LIKE '$like_term'";
    $where_conditions[] = "ci.car_type LIKE '$like_term'";
    $where_conditions[] = "ci.color LIKE '$like_term'";
    $where_conditions[] = "ci.transmission LIKE '$like_term'";
    $where_conditions[] = "ci.fuel_type LIKE '$like_term'";
    $where_conditions[] = "ci.terrain LIKE '$like_term'";

    // 2. Yes/No flag fields → match human-readable keywords
    $yes_no_mapping = [
        'driver'        => "cr.with_driver = 1",
        'with driver'   => "cr.with_driver = 1",
        'self drive'    => "cr.with_driver = 0",
        'self-drive'    => "cr.with_driver = 0",
        'aircon'        => "ci.aircon = 'Y'",
        'airconditioned' => "ci.aircon = 'Y'",
        'ac'            => "ci.aircon = 'Y'",
        'budget'        => "ci.budget_friendly = 'Y'",
        'cheap'         => "ci.budget_friendly = 'Y'",
        'affordable'    => "ci.budget_friendly = 'Y'",
        'wide compartment' => "ci.wide_compartment = 'Y'",
        'spacious'      => "ci.wide_compartment = 'Y'",
        'big trunk'     => "ci.wide_compartment = 'Y'",
        'child seat'    => "ci.child_seat = 'Y'",
        'baby seat'     => "ci.child_seat = 'Y'",
        'wide leg room' => "ci.wide_leg_room = 'Y'",
        'legroom'       => "ci.wide_leg_room = 'Y'",
        'leg room'      => "ci.wide_leg_room = 'Y'",
        'special needs' => "ci.special_needs_friendly = 'Y'",
        'pwd'           => "ci.special_needs_friendly = 'Y'",
        'senior'        => "ci.special_needs_friendly = 'Y'",
        'accessible'    => "ci.special_needs_friendly = 'Y'"
    ];

    foreach ($yes_no_mapping as $keyword => $condition) {
        if (strpos($lower_search, $keyword) !== false) {
            $where_conditions[] = $condition;
        }
    }

    // 3. Car Attributes (text + JSON)
    $where_conditions[] = "ca.seating_capacity LIKE '$like_term'";
    $where_conditions[] = "ca.terrain_compatibility LIKE '$like_term'";
    $where_conditions[] = "ca.ground_clearance LIKE '$like_term'";
    $where_conditions[] = "ca.cargo_space LIKE '$like_term'";
    $where_conditions[] = "ca.body_size LIKE '$like_term'";
    $where_conditions[] = "ca.compact LIKE '$like_term'";

    // JSON trip_purpose search
    $where_conditions[] = "JSON_SEARCH(LOWER(ca.trip_purpose), 'one', '%" . strtolower($json_term) . "%') IS NOT NULL";

    // Reviews search
    $where_conditions[] = "EXISTS (
        SELECT 1 FROM reviews r
        JOIN bookings b ON r.booking_id = b.id
        WHERE b.car_id = ci.id AND LOWER(r.comment) LIKE '$like_term'
    )";

    // Final WHERE clause
    $where_clause = implode(' OR ', $where_conditions);

    // Count total results
    $count_sql = "
        SELECT COUNT(DISTINCT ci.id) as total
        FROM car_inventory ci
        LEFT JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
        LEFT JOIN car_rates cr ON ci.id = cr.car_id
        LEFT JOIN car_attributes ca ON ci.id = ca.car_id
        WHERE $where_clause
    ";
    $count_result = $conn->query($count_sql);
    $total_search_results = $count_result ? $count_result->fetch_assoc()['total'] : 0;

    // Pagination
    $results_per_page = 9;
    $total_search_pages = ceil($total_search_results / $results_per_page);
    $current_search_page = max(1, min($total_search_pages, (int)($_GET['search_page'] ?? 1)));
    $offset = ($current_search_page - 1) * $results_per_page;

    // FIX: Main search query — added ALL individual rate columns
    $search_sql = "
        SELECT
            ci.*,
            sa.shop_name,
            COUNT(b.id) as booking_count,
            COALESCE(cr.rate_inside_zambales_12hrs,  3000) AS rate,
            cr.with_driver,
            COALESCE(cr.driver_price_12hrs,           500) AS driver_price,
            /* ── ALL RATE COLUMNS (for booking modal hidden inputs) ── */
            COALESCE(cr.rate_inside_zambales_12hrs,  3000) AS rate_inside_zambales_12hrs,
            COALESCE(cr.rate_inside_zambales_24hrs,  5000) AS rate_inside_zambales_24hrs,
            COALESCE(cr.rate_outside_zambales_12hrs, 3500) AS rate_outside_zambales_12hrs,
            COALESCE(cr.rate_outside_zambales_24hrs, 6000) AS rate_outside_zambales_24hrs,
            COALESCE(cr.rate_baguio_12hrs,           4000) AS rate_baguio_12hrs,
            COALESCE(cr.rate_baguio_24hrs,           7000) AS rate_baguio_24hrs,
            COALESCE(cr.driver_price_12hrs,           500) AS driver_price_12hrs,
            COALESCE(cr.driver_price_24hrs,           800) AS driver_price_24hrs,
            ca.seating_capacity,
            ca.terrain_compatibility,
            ca.trip_purpose,
            ca.ground_clearance,
            ca.cargo_space,
            ca.body_size,
            ca.compact
        FROM car_inventory ci
        LEFT JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
        LEFT JOIN bookings b ON ci.id = b.car_id
        LEFT JOIN car_rates cr ON ci.id = cr.car_id
        LEFT JOIN car_attributes ca ON ci.id = ca.car_id
        WHERE $where_clause
        GROUP BY ci.id
        ORDER BY ci.id DESC
        LIMIT $offset, $results_per_page
    ";

    $search_results = $conn->query($search_sql);
    $result = $search_results;
    $total_pages = $total_search_pages;
    $current_page = $current_search_page;
}
?>
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home - CarZam</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.10.5/font/bootstrap-icons.min.css"
        rel="stylesheet">
    <style>
        :root {
            --white: #FFFFFF;
            --dark-gray: #242424;
            --light-blue: #5C7C89;
            --medium-blue: #1F4959;
            --dark-blue: #011425;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--dark-gray);
            background-color: #f5f7fa;
        }

        .navbar {
            background-color: var(--dark-blue);
            padding: 12px 0;
        }

        .navbar-brand {
            font-weight: 700;
            color: var(--white) !important;
            font-size: 1.5rem;
        }

        .nav-link {
            color: var(--white) !important;
            margin-right: 15px;
            transition: all 0.3s;
            font-weight: 500;
        }

        .nav-link:hover,
        .nav-link.active {
            color: var(--light-blue) !important;
        }

        .nav-link .bi {
            margin-right: 6px;
        }

        .user-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 8px;
        }

        .dropdown-menu {
            border: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }

        .dropdown-item:active {
            background-color: var(--medium-blue);
        }

        .btn-primary {
            background-color: var(--medium-blue);
            border-color: var(--medium-blue);
        }

        .btn-primary:hover {
            background-color: var(--dark-blue);
            border-color: var(--dark-blue);
        }

        .btn-outline-primary {
            color: var(--medium-blue);
            border-color: var(--medium-blue);
        }

        .btn-outline-primary:hover {
            background-color: var(--medium-blue);
            border-color: var(--medium-blue);
            color: var(--white);
        }

        .search-container {
            background: white;
            padding: 40px 0;
            color: var(--white);
        }

        .search-form {
            background-color: var(--white);
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            color: var(--dark-gray);
        }

        .form-select:focus,
        .form-control:focus {
            border-color: var(--medium-blue);
            box-shadow: 0 0 0 0.25rem rgba(92, 124, 137, 0.25);
        }

        .car-section {
            padding: 50px 0;
        }

        .section-title {
            position: relative;
            margin-bottom: 30px;
            padding-bottom: 15px;
            color: black;
        }

        .section-title::after {
            content: '';
            position: absolute;
            left: 0;
            bottom: 0;
            width: 60px;
            height: 4px;
            background-color: var(--medium-blue);
        }

        .filter-bar {
            background-color: var(--white);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .car-img-top {
            height: 350px;
            width: 100%;
            object-fit: cover;
            border-radius: 10px 10px 0 0;
            cursor: pointer;
            transition: transform 0.3s;
        }

        .car-img-top:hover {
            transform: scale(1.05);
        }

        .car-card {
            background-color: var(--white);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s;
            height: 100%;
            border: none;
            display: flex;
            flex-direction: column;
        }

        .car-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
        }

        .car-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background-color: rgba(1, 20, 37, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 500;
        }

        .car-title {
            font-weight: 700;
            margin-bottom: 5px;
        }

        .car-price {
            color: var(--medium-blue);
            font-weight: 700;
            font-size: 1.25rem;
        }

        .car-features {
            padding: 0;
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 15px;
        }

        .car-features li {
            display: flex;
            align-items: center;
            font-size: 0.9rem;
            color: #6c757d;
        }

        .car-features i {
            margin-right: 5px;
            color: var(--medium-blue);
        }

        .car-rating {
            color: #ffc107;
        }

        .pagination .page-item.active .page-link {
            background-color: var(--medium-blue);
            border-color: var(--medium-blue);
            color: white;
        }

        .pagination .page-link {
            color: var(--medium-blue);
        }

        .pagination .page-link:focus {
            box-shadow: 0 0 0 0.25rem rgba(92, 124, 137, 0.25);
        }

        .notification-badge {
            position: absolute;
            top: 0;
            right: 0;
            background-color: #dc3545;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        footer {
            background-color: var(--dark-blue);
            color: var(--white);
            padding: 60px 0 20px;
        }

        .footer-heading {
            color: var(--light-blue);
            margin-bottom: 20px;
        }

        .footer-link {
            color: var(--white);
            text-decoration: none;
            display: block;
            margin-bottom: 10px;
            transition: all 0.3s;
        }

        .footer-link:hover {
            color: var(--light-blue);
        }

        .social-icon {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: var(--medium-blue);
            color: var(--white);
            margin-right: 10px;
            transition: all 0.3s;
        }

        .social-icon:hover {
            background-color: var(--light-blue);
            transform: translateY(-3px);
        }
    </style>
</head>

<body>
    <!-- Navbar -->

    <nav class="navbar navbar-expand-lg navbar-dark sticky-top">
        <div class="container">
            <a class="navbar-brand" href="#">CarZam</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"
                aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link active" href="index.php"><i class="bi bi-house"></i> Home</a>
                    </li>

                    <li class="nav-item">
                        <a class="nav-link" href="shop.php"><i class="bi bi-shop"></i> Shops</a>
                    </li>
                </ul>
                <ul class="navbar-nav">
                    <li class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle d-flex align-items-center" href="#" id="navbarDropdown"
                            role="button" data-bs-toggle="dropdown" aria-expanded="false">
                            <img src="user.png" alt="Profile" class="user-avatar">
                            <span><?php echo htmlspecialchars($user_name); ?></span>
                        </a>
                        <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="navbarDropdown">
                            <li><a class="dropdown-item" href="Profile.php"><i class="bi bi-person"></i> My Profile</a></li>
                            <li><a class="dropdown-item" href="BookingHistory.php"><i class="bi bi-clock-history"></i> Booking History</a></li>
                            <li><a class="dropdown-item" href="chat.php"><i class="bi bi-chat-dots"></i> Chats</a></li>
                            <li>
                                <hr class="dropdown-divider">
                            </li>
                            <li><a class="dropdown-item text-danger" href="../index.php"><i class="bi bi-box-arrow-right"></i> Log Out</a></li>
                        </ul>
                    </li>
                </ul>

            </div>
        </div>
    </nav>

    <!-- Bootstrap JS Bundle with Popper -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>


    <!-- SEARCH SECTION -->
    <section class="search-container py-5">
        <div class="container">
            <h2 class="section-title mb-4 text-center">CARZAM</h2>

            <!-- Search Bar Section -->
            <div class="search-section mb-4">
                <div class="row g-3 align-items-center">
                    <div class="col-md-8">
                        <form method="GET" action="" class="d-flex gap-2" id="searchForm">
                            <div class="flex-grow-1 position-relative">
                                <input type="text"
                                    class="form-control search-input"
                                    name="search"
                                    id="searchInput"
                                    placeholder="Search cars by name, shop, type, color, transmission, fuel, terrain..."
                                    value="<?= htmlspecialchars($search_query) ?>"
                                    autocomplete="off">
                                <div class="position-absolute end-0 top-50 translate-middle-y me-3 search-icon">
                                    <i class="bi bi-search text-muted"></i>
                                </div>
                                <?php if ($search_performed && $search_query): ?>
                                    <div class="position-absolute start-0 top-50 translate-middle-y ms-3 search-tag">
                                        <span class="badge bg-primary"><?= htmlspecialchars($search_query) ?> <i class="bi bi-x-circle ms-1" onclick="clearSearch()"></i></span>
                                    </div>
                                <?php endif; ?>
                            </div>
                            <button type="submit" class="btn btn-primary px-4" id="searchBtn">
                                <i class="bi bi-search me-1"></i>Search
                            </button>
                            <?php if ($search_performed): ?>
                                <a href="?<?= $_SERVER['QUERY_STRING'] ? str_replace(['&search=', 'search='], '', $_SERVER['QUERY_STRING']) : '' ?>"
                                    class="btn btn-outline-secondary" id="clearSearchBtn">
                                    <i class="bi bi-x-circle me-1"></i>Clear
                                </a>
                            <?php endif; ?>
                        </form>
                    </div>
                    <div class="col-md-4">
                        <?php if ($search_performed): ?>
                            <div class="search-results-info text-end">
                                <small class="text-muted">
                                    <i class="bi bi-info-circle me-1"></i>
                                    Found <strong><?= $total_search_results ?></strong> results for "<?= htmlspecialchars($search_query) ?>"
                                    <?php if ($total_search_pages > 1): ?>
                                        | Page <?= $current_search_page ?> of <?= $total_search_pages ?>
                                    <?php endif; ?>
                                </small>
                            </div>
                        <?php endif; ?>
                    </div>
                </div>
            </div>
            <div class="card shadow-lg border-0 mt-4 card-shadow">
                <div class="card-body p-4">
                    <form id="findCarsForm">
                        <div class="row g-3">
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label fw-bold">Capacity</label>
                                <select class="form-select" name="capacity" id="capacity" required>
                                    <option value="">Select Capacity</option>
                                    <option value="4">4 Seats</option>
                                    <option value="5">5 Seats</option>
                                    <option value="7">7 Seats</option>
                                    <option value="12">12 Seats</option>
                                    <option value="15">15 Seats</option>
                                    <option value="21">21 Seats</option>
                                </select>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label fw-bold">Trip Purpose</label>
                                <input type="text" class="form-control" name="trip_purpose" id="trip_purpose"
                                       placeholder="e.g. Family Trip, Beach Trip..." required>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label fw-bold">Duration</label>
                                <select class="form-select" name="duration" id="duration" required>
                                    <option value="">Select Duration</option>
                                    <option value="1">12 Hours</option>
                                    <option value="2">24 Hours</option>
                                    <option value="7">1 Week</option>
                                    <option value="30">1 Month</option>
                                </select>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label fw-bold">Budget (₱)</label>
                                <input type="number" class="form-control" name="budget" id="budget" min="1000" max="50000" step="500" value="0" required>
                            </div>
                            <div class="col-md-12 col-lg-4">
                                <label class="form-label fw-bold">Additional Needs</label>
                                <textarea class="form-control" name="inquiry" id="inquiry" rows="2"
                                    placeholder="e.g. diesel, automatic, child seat, red color, strong AC..."></textarea>
                                <small class="text-muted">AI reads every word</small>
                            </div>
                            <div class="col-12 text-center mt-4">
                                <button type="submit" class="btn btn-primary btn-lg px-5 shadow">
                                    AI Predict Cars
                                </button>
                                <button type="button" id="resetFilters" class="btn btn-outline-danger ms-3">
                                    Reset
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body text-center py-4">
                            <div id="aiStatus" class="d-none">
                                <h4 class="text-success fw-bold">100% Pure Machine Learning Active</h4>
                                <p class="text-muted" id="modelInfo"></p>
                            </div>
                            <div id="aiLoading" class="d-none">
                                <div class="spinner-border text-primary mb-3" style="width: 3rem; height: 3rem;"></div>
                                <h5 class="text-primary">AI is thinking...</h5>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    <!-- RESULTS SECTION -->
    <section class="car-section py-5 bg-light" id="resultsSection" style="display: none;">
        <div class="container">
            <div class="row mb-4 align-items-center">
                <!-- REMOVED: AI Top X Recommendations header + confidence alert -->

            </div>

            <div class="card border-success mb-4" id="aiExplanation" style="display: none;">
                <div class="card-body">
                    <h5 class="card-title text-success">Why We Chose These Cars</h5>
                    <p class="card-text fs-5" id="explanationText"></p>
                </div>
            </div>

            <!-- REMOVED: Entire aiDecisionSummary card removed -->

            <div class="row g-4" id="predictedCarsContainer"></div>
        </div>
    </section>
    <!-- GALLERY MODAL -->
    <div class="modal fade" id="galleryModal" tabindex="-1">
        <div class="modal-dialog modal-lg modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Car Gallery</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body text-center">
                    <div id="galleryCarousel" class="carousel slide">
                        <div class="carousel-inner" id="galleryImages"></div>
                        <button class="carousel-control-prev" type="button" data-bs-target="#galleryCarousel" data-bs-slide="prev">
                            <span class="carousel-control-prev-icon"></span>
                        </button>
                        <button class="carousel-control-next" type="button" data-bs-target="#galleryCarousel" data-bs-slide="next">
                            <span class="carousel-control-next-icon"></span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Trip Purposes (unchanged)
        const tripPurposesByCapacity = {
            '4': ['Cargo Delivery', 'Off-road Trip'],
            '5': ['Point-to-point', 'Daily Use', 'Travel', 'Family Trip', 'Off-road Trip'],
            '7': ['Family Trip', 'Airport Run', 'Leisure', 'Off-road Trip'],
            '12': ['Shuttle Service', 'Family Trip', 'School Trip', 'Airport Drop-off'],
            '15': ['Shuttle Service', 'Family Trip', 'School Trip', 'Airport Drop-off'],
            '21': ['Group Transport', 'Cargo Delivery']
        };

        document.getElementById('capacity').addEventListener('change', function() {
            const sel = document.getElementById('trip_purpose');
            sel.innerHTML = '<option value="">Select Purpose</option>';
            sel.disabled = true;
            if (this.value && tripPurposesByCapacity[this.value]) {
                tripPurposesByCapacity[this.value].forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p;
                    opt.textContent = p;
                    sel.appendChild(opt);
                });
                sel.disabled = false;
            }
        });

        document.getElementById('findCarsForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const payload = {
                capacity: parseInt(document.getElementById('capacity').value),
                trip_purpose: document.getElementById('trip_purpose').value,
                duration_days: parseInt(document.getElementById('duration').value),
                budget: parseFloat(document.getElementById('budget').value),
                inquiry: document.getElementById('inquiry').value.trim()
            };
            const btn = this.querySelector('button[type="submit"]');
            const orig = btn.innerHTML;
            btn.innerHTML = 'Loading...';
            btn.disabled = true;
            document.getElementById('normalCarList').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('predictedCarsContainer').innerHTML = '<div class="col-12 text-center py-5"><div class="spinner-border text-primary" style="width: 4rem; height: 4rem;"></div></div>';
            try {
                const res = await fetch('https://carzam-api-8trx.onrender.com/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                renderRecommendedCars(data.recommendations || [], payload);
            } catch (err) {
                document.getElementById('predictedCarsContainer').innerHTML = `
            <div class="col-12 text-center py-5">
                <h4 class="text-danger">AI Server Not Responding</h4>
                <p>Make sure your Python server is running on port 5000</p>
                <button class="btn btn-primary" onclick="window.location.reload()">Try Again</button>
            </div>`;
                console.error(err);
            } finally {
                btn.innerHTML = orig;
                btn.disabled = false;
            }
        });

        function renderRecommendedCars(cars, userInput) {
            const container = document.getElementById('predictedCarsContainer');
            container.innerHTML = '';
            const now = new Date().toISOString().slice(0, 16);
            const userName = <?= json_encode($user_name) ?>;
            const is24hrs = userInput.duration_days === 2;
            const durationText = is24hrs ? '24 Hours' : '12 Hours';

            if (cars.length === 0) {
                container.innerHTML = '<div class="col-12 text-center py-5"><h5>No cars match your needs right now.</h5></div>';
                return;
            }

            cars.forEach(car => {
                const carId = car.car_id;
                const baseRate12 = Number(car.rate) || 3000;
                const displayedRate = baseRate12;
                const rateText = '/12hrs';
                const totalBookings = car.booking_count || 0;
                const inquiry = userInput.inquiry.toLowerCase();

                let reasons = [
                    `${userInput.capacity} seater as requested`,
                    `Perfect for ${userInput.trip_purpose}`
                ];

                if (inquiry.includes('diesel')) reasons.push('Diesel engine');
                if (inquiry.includes('automatic') || inquiry.includes('auto')) reasons.push('Automatic transmission');
                if (inquiry.includes('red')) reasons.push('Red color');
                if (inquiry.includes('child') || inquiry.includes('baby')) reasons.push('Child seat ready');

                const bookingModal = `
        <div class="modal fade" id="bookingModal${carId}" tabindex="-1">
            <div class="modal-dialog"><div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Book ${car.car_name}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
<form method="post" action="">
    <input type="hidden" name="car_id" value="<?php echo $row['id']; ?>">
    <input type="hidden" name="duration" value="<?php echo htmlspecialchars($user_inputs['duration'] ?? '12hrs'); ?>">
    
    <!-- Hidden Rate Fields (unchanged) -->
    <input type="hidden" id="rate_inside_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_inside_zambales_12hrs'] ?? 3000; ?>">
    <input type="hidden" id="rate_inside_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_inside_zambales_24hrs'] ?? 5000; ?>">
    <input type="hidden" id="rate_outside_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_outside_zambales_12hrs'] ?? 3500; ?>">
    <input type="hidden" id="rate_outside_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_outside_zambales_24hrs'] ?? 6000; ?>">
    <input type="hidden" id="rate_baguio_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_baguio_12hrs'] ?? 4000; ?>">
    <input type="hidden" id="rate_baguio_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_baguio_24hrs'] ?? 7000; ?>">
    <input type="hidden" id="driver_price_12_<?php echo $row['id']; ?>" value="<?php echo $row['driver_price_12hrs'] ?? 500; ?>">
    <input type="hidden" id="driver_price_24_<?php echo $row['id']; ?>" value="<?php echo $row['driver_price_24hrs'] ?? 800; ?>">
    <input type="hidden" id="with_driver_<?php echo $row['id']; ?>" value="<?php echo $row['with_driver'] ? 'true' : 'false'; ?>">
    <input type="hidden" id="stakeholder_id_<?php echo $row['id']; ?>" value="<?php echo $row['stakeholder_id']; ?>">

    <!-- Full Name -->
    <div class="mb-3">
        <label class="form-label">Full Name</label>
        <input type="text" class="form-control" value="<?php echo htmlspecialchars($user_name); ?>" disabled>
    </div>

    <!-- Address -->
    <div class="mb-3">
        <label for="address<?php echo $row['id']; ?>" class="form-label">Pickup Address <span class="text-danger">*</span></label>
        <input type="text" class="form-control" id="address<?php echo $row['id']; ?>" 
               name="address" placeholder="Enter your pickup location" required>
    </div>

    <!-- Destination -->
    <div class="mb-3">
        <label for="destination<?php echo $row['id']; ?>" class="form-label">Destination <span class="text-danger">*</span></label>
        <select class="form-select" id="destination<?php echo $row['id']; ?>" 
                name="destination" required onchange="updateTotalPrice(<?php echo $row['id']; ?>)">
            <option value="">Select Destination</option>
            <?php foreach ($destinations as $dest): ?>
                <option value="<?php echo htmlspecialchars($dest); ?>">
                    <?php echo htmlspecialchars($dest); ?>
                </option>
            <?php endforeach; ?>
        </select>
    </div>

    <!-- Trip Purpose — dropdown -->
    <div class="mb-3">
        <label for="trip_purpose_ai_${carId}" class="form-label">Trip Purpose <span class="text-danger">*</span></label>
        <select class="form-select" id="trip_purpose_ai_${carId}" name="trip_purpose" required>
            <option value="">Select Purpose</option>
            <option value="Family Trip">Family Trip</option>
            <option value="Business">Business</option>
            <option value="Wedding">Wedding</option>
            <option value="Adventure/Outdoor">Adventure/Outdoor</option>
            <option value="Beach Trip">Beach Trip</option>
            <option value="Mountain Trip">Mountain Trip</option>
            <option value="City Tour">City Tour</option>
            <option value="Airport Transfer">Airport Transfer</option>
            <option value="Shuttle Service">Shuttle Service</option>
            <option value="School Trip">School Trip</option>
            <option value="Special Event">Special Event</option>
            <option value="Off-road Trip">Off-road Trip</option>
            <option value="Others">Others</option>
        </select>
    </div>

    <!-- NEW: Number of Seaters -->
    <div class="mb-3">
        <label for="number_of_seaters<?php echo $row['id']; ?>" class="form-label">
            Number of Passengers <span class="text-danger">*</span>
            <small class="text-muted">(Max: <?php echo $row['capacity']; ?> seats)</small>
        </label>
        <input type="number" class="form-control" 
               id="number_of_seaters<?php echo $row['id']; ?>" 
               name="number_of_seaters" 
               min="1" 
               max="<?php echo $row['capacity']; ?>" 
               placeholder="Enter number of passengers" 
               required>
    </div>

    <!-- Booking Status -->
    <div class="mb-3">
        <label class="form-label">Booking Status</label>
        <input type="text" class="form-control" value="Pending" disabled>
    </div>

    <!-- Payment Status -->
    <div class="mb-3">
        <label class="form-label">Payment Status</label>
        <input type="text" class="form-control" value="Unpaid" disabled>
    </div>

    <!-- Booking Date & Time -->
    <div class="mb-3">
        <label for="booking_date<?php echo $row['id']; ?>" class="form-label">Booking Date & Time <span class="text-danger">*</span></label>
        <input type="datetime-local" class="form-control" 
               id="booking_date<?php echo $row['id']; ?>" 
               name="booking_date" 
               required 
               min="<?php echo date('Y-m-d\TH:i'); ?>" 
               value="<?php echo date('Y-m-d\TH:i'); ?>" 
               onchange="updateTotalPrice(<?php echo $row['id']; ?>); validateBookingDate(<?php echo $row['id']; ?>)">
        <small id="booking_date_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
            Booking date must be in the future.
        </small>
    </div>

    <!-- Return Date & Time -->
    <div class="mb-3">
        <label for="return_date<?php echo $row['id']; ?>" class="form-label">Return Date & Time <span class="text-danger">*</span></label>
        <input type="datetime-local" class="form-control" 
               id="return_date<?php echo $row['id']; ?>" 
               name="return_date" 
               required 
               min="<?php echo date('Y-m-d\TH:i'); ?>" 
               onchange="updateTotalPrice(<?php echo $row['id']; ?>); validateReturnDate(<?php echo $row['id']; ?>)">
        <small id="return_date_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
            Return date must be after booking date.
        </small>
        <small id="mk_rental_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
            MK Car Rental requires minimum 48 hours.
        </small>
    </div>

    <!-- Total Price -->
    <div class="mb-3">
        <label class="form-label">Total Price</label>
        <input type="text" class="form-control fw-bold text-primary" 
               id="total_price<?php echo $row['id']; ?>" 
               value="₱0.00" disabled>
    </div>

    <div class="mb-3">
        <small class="text-muted">• Overstay by 1 hour = extra full rate charge</small><br>
        <small class="text-muted">• No refund for early return</small>
        <?php if ($row['stakeholder_id'] == 7): ?>
            <br><small class="text-warning fw-bold">• MK Car Rental: Minimum 2 days (48 hours)</small>
        <?php endif; ?>
    </div>

    <hr class="my-4">

    <!-- RENTAL AGREEMENT -->
    <div class="mb-3">
        <h6 class="fw-bold text-primary mb-3">
            <i class="bi bi-file-text-fill me-2"></i>Car Rental Contract Agreement
        </h6>
        
        <div class="border rounded p-3 mb-3" style="max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
            <p class="small mb-2">
                <strong>This Car Rental Contract Agreement</strong> ("Agreement") is entered into by and between the 
                Car Rental Provider (hereinafter referred to as the "Company") and the Customer/Renter 
                (hereinafter referred to as the "Customer"). By confirming the booking and using the rented vehicle, 
                the Customer agrees to all the terms and conditions stated herein.
            </p>

            <p class="small mb-2"><strong>1. AGREEMENT TO PAY RENTAL FEES</strong></p>
            <p class="small mb-2">
                The Customer agrees to pay the total rental amount as displayed, quoted, or calculated by the Company 
                at the time of booking or vehicle release. This amount includes the agreed rental duration, vehicle type, 
                and other applicable charges disclosed prior to confirmation. The Customer acknowledges that failure to 
                pay the total agreed amount may result in penalties, additional charges, or legal action in accordance 
                with applicable laws.
            </p>

            <p class="small mb-2"><strong>2. ADDITIONAL CHARGES FOR EXCEEDED RENTAL DURATION</strong></p>
            <p class="small mb-2">
                The Customer agrees that if the rented vehicle is returned beyond the agreed rental duration, additional 
                charges shall apply. These charges will be calculated based on the Company's prevailing rate (hourly or daily) 
                and shall be paid immediately upon vehicle return. Any delay without prior notice or approval from the Company 
                may be subject to extra penalties.
            </p>

            <p class="small mb-2"><strong>3. DAMAGE, LOSS, AND REPAIR LIABILITY</strong></p>
            <p class="small mb-2">
                The Customer agrees to be fully responsible for any damage, loss, or theft of the rented vehicle during 
                the rental period, except for normal wear and tear. If the vehicle is returned with damages, the Customer 
                agrees to: (1) Pay the assessed damage repair cost, and/or (2) Cover replacement costs for missing or 
                damaged parts, and/or (3) Pay any applicable insurance excess or damage fee.
            </p>

            <p class="small mb-2"><strong>4. DRIVER'S LICENSE AND REQUIREMENTS</strong></p>
            <p class="small mb-2">
                The Customer confirms and warrants that: (1) The designated driver holds a valid and non-expired driver's 
                license appropriate for the vehicle type rented; (2) All documents and requirements submitted to the Company 
                are true, accurate, and valid; (3) Only authorized and licensed drivers approved by the Company will operate 
                the rented vehicle.
            </p>

            <p class="small mb-2"><strong>5. SAFE AND RESPONSIBLE USE OF VEHICLE</strong></p>
            <p class="small mb-2">
                The Customer agrees to operate the rented vehicle in a safe, careful, and lawful manner at all times. 
                The Customer shall: (1) Follow all traffic laws and regulations; (2) Avoid reckless, negligent, or dangerous 
                driving; (3) Not use the vehicle for illegal activities, racing, or prohibited purposes; (4) Not drive under 
                the influence of alcohol, drugs, or any impairing substances. Any violation resulting in fines, penalties, 
                or damage shall be the sole responsibility of the Customer.
            </p>

            <p class="small mb-2"><strong>6. ASSUMPTION OF RESPONSIBILITY</strong></p>
            <p class="small mb-2">
                The Customer assumes full responsibility for the vehicle from the time it is released by the Company until 
                it is properly returned and accepted by the Company.
            </p>

            <p class="small mb-0"><strong>7. ACCEPTANCE OF TERMS</strong></p>
            <p class="small mb-0">
                By signing this Agreement or confirming the rental booking, the Customer acknowledges that they have read, 
                understood, and agreed to all the terms and conditions stated herein.
            </p>
        </div>

        <!-- Agreement Checkbox -->
        <div class="form-check mb-3 p-3 border rounded" style="background-color: #fff3cd;">
            <input type="checkbox" class="form-check-input" 
                   id="terms_accepted_<?= $row['id'] ?>" 
                   name="terms_accepted"
                   required
                   onchange="toggleSubmitButton(<?= $row['id'] ?>)">
            <label class="form-check-label fw-bold" for="terms_accepted_<?= $row['id'] ?>">
                <i class="bi bi-check-circle text-success me-1"></i>
                I have read and agree to all the terms and conditions of this Car Rental Contract Agreement
            </label>
        </div>

        <!-- Price Agreement Checkbox -->
        <div class="form-check mb-3 p-3 border rounded" style="background-color: #d1ecf1;">
            <input type="checkbox" class="form-check-input" 
                   id="agree_price_<?= $row['id'] ?>" 
                   required
                   onchange="toggleSubmitButton(<?= $row['id'] ?>)">
            <label class="form-check-label" for="agree_price_<?= $row['id'] ?>">
                I agree to pay <strong id="final_price_display_<?= $row['id'] ?>">₱0.00</strong> for this rental.<br>
                <small class="text-muted">• No refund for early return • Overstay = full extra day charge</small>
            </label>
        </div>
    </div>

    <!-- Submit Button -->
    <button type="submit" name="book_car" 
            class="btn btn-primary w-100 py-2 fw-bold" 
            id="submit_btn_<?= $row['id'] ?>"
            disabled>
        <i class="bi bi-check-circle me-2"></i>Confirm Booking
    </button>

    <p class="text-center text-muted small mt-3 mb-0">
        <i class="bi bi-shield-check me-1"></i>Your booking will be reviewed and confirmed by the car owner
    </p>
</form>
                </div>
            </div></div>
        </div>`;

                const cardHtml = `
        <div class="col-md-6 col-lg-4">
            ${bookingModal}
            <div class="card car-card h-100 shadow-sm">
                <div class="position-relative">
                    <img src="../stakeholders/uploads/cars/${car.image || 'PLACEHOLDER.png'}"
                         class="car-img-top gallery-trigger"
                         alt="${car.car_name}"
                         onerror="this.src='../stakeholders/uploads/cars/PLACEHOLDER.png'">
                    <div class="position-absolute bottom-0 start-0 m-3">
                        <span class="badge bg-dark opacity-75"><i class="bi bi-images"></i> 1 photo</span>
                    </div>
                </div>
                <div class="card-body d-flex flex-column">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="mb-0">${car.car_name}</h5>
                        <span class="badge bg-secondary">${car.shop_name}</span>
                    </div>
                    <div class="d-flex justify-content-end mb-3">
                        <span class="car-price">₱${displayedRate.toLocaleString()}<small class="text-muted">${rateText}</small></span>
                    </div>
                    <div class="d-flex justify-content-between mb-3">
                        <small class="text-muted">
                            Booked <strong>${totalBookings}</strong> time${totalBookings !== 1 ? 's' : ''}
                        </small>
                    </div>
                    <ul class="car-features list-unstyled" style="min-height: 280px;">
                        <li><i class="bi bi-people text-primary me-2"></i> ${car.capacity} Seats</li>
                        <li><i class="bi bi-paint-bucket text-info me-2"></i> ${car.color || 'Silver'}</li>
                        <li><i class="bi bi-gear text-secondary me-2"></i> ${car.transmission || 'Auto'}</li>
                        <li><i class="bi bi-fuel-pump text-warning me-2"></i> ${car.fuel_type || 'Gasoline'}</li>
                        <li><i class="bi bi-car-front text-success me-2"></i> ${car.car_type}</li>
                        ${car.special_needs_friendly === 'Y' ? '<li><i class="bi bi-heart-fill text-danger me-2"></i> PWD & Senior Friendly</li>' : ''}
                        ${car.child_seat === 'Y' ? '<li><i class="bi bi-baby-carriage text-pink me-2"></i> Child Seat Available</li>' : ''}
                        ${car.wide_leg_room === 'Y' ? '<li><i class="bi bi-arrows-expand text-indigo me-2"></i> Wide Leg Room</li>' : ''}
                        ${car.terrain ? `<li><i class="bi bi-map text-success me-2"></i> ${car.terrain} Terrain</li>` : '<li><i class="bi bi-map text-success me-2"></i> Mixed Terrain</li>'}
                        ${car.aircon === 'Y' ? '<li><i class="bi bi-fan text-info me-2"></i> Air Conditioned</li>' : ''}
                        ${car.wide_compartment === 'Y' ? '<li><i class="bi bi-box text-primary me-2"></i> Wide Compartment</li>' : ''}
                        <li><i class="bi bi-person text-dark me-2"></i> Self-Drive</li>
                    </ul>
                    <div class="alert alert-light py-2 small border-start border-success border-4 mb-3" style="height: 65px; display: flex; flex-direction: column; justify-content: center;">
                        <div>
                            <strong>Why this car?</strong><br>
                            ${reasons.join(' • ')}
                        </div>
                    </div>
                    <div class="d-grid mt-auto gap-2">
                        <button class="btn btn-primary btn-lg" data-bs-toggle="modal" data-bs-target="#bookingModal${carId}">Book Now</button>
                        <button type="button" class="btn btn-outline-primary view-reviews-btn"
                            data-bs-toggle="modal" data-bs-target="#reviewsModal"
                            data-car-id="${carId}" data-car-name="${car.car_name}">
                            View Reviews
                        </button>
                    </div>
                </div>
            </div>
        </div>`;

                container.innerHTML += cardHtml;
            });

            document.getElementById('resultsSection').scrollIntoView({
                behavior: 'smooth'
            });
        }

        document.getElementById('resetFilters').addEventListener('click', () => {
            window.location.reload();
        });

        // Review button handler - using event delegation for dynamically added elements
        document.addEventListener('click', function(e) {
            if (e.target && e.target.classList.contains('view-reviews-btn')) {
                const carId = e.target.getAttribute('data-car-id');
                const carName = e.target.getAttribute('data-car-name');

                // Update modal title
                const modalCarName = document.getElementById('modalCarName');
                if (modalCarName) {
                    modalCarName.textContent = carName;
                }

                // Fetch reviews via AJAX
                const reviewsContent = document.getElementById('reviewsContent');
                if (reviewsContent) {
                    reviewsContent.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary"></div><p class="mt-2">Loading reviews...</p></div>';

                    fetch('get_reviews.php', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/x-www-form-urlencoded',
                            },
                            body: 'car_id=' + encodeURIComponent(carId)
                        })
                        .then(response => response.text())
                        .then(data => {
                            reviewsContent.innerHTML = data;
                        })
                        .catch(error => {
                            reviewsContent.innerHTML = '<p class="text-danger">Error loading reviews. Please try again later.</p>';
                            console.error('Error fetching reviews:', error);
                        });
                }
            }
        });
    </script>
    <section class="car-section" id="normalCarList">
        <div class="container">
            <?php if (isset($_SESSION['booking_success'])): ?>
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    <?= htmlspecialchars($_SESSION['booking_success']) ?>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
                <?php unset($_SESSION['booking_success']); ?>
            <?php endif; ?>
            <?php if (isset($_SESSION['booking_error'])): ?>
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <?= htmlspecialchars($_SESSION['booking_error']) ?>
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
                <?php unset($_SESSION['booking_error']); ?>
            <?php endif; ?>


            <h2 class="section-title mb-4">
                <?php if ($search_performed): ?>
                    Search Results for "<?= htmlspecialchars($search_query) ?>"
                <?php else: ?>
                    Available Cars
                <?php endif; ?>
            </h2>

            <div class="row g-4">
                <?php
                if ($result && $result->num_rows > 0):
                    while ($row = $result->fetch_assoc()):
                        $total_price = $row['rate'] + ($row['with_driver'] ? $row['driver_price'] : 0);
                ?>
                        <div class="col-md-6 col-lg-4">
                            <!-- Booking Modal -->
                            <div class="modal fade" id="bookingModal<?php echo $row['id']; ?>" tabindex="-1" aria-labelledby="bookingModalLabel<?php echo $row['id']; ?>" aria-hidden="true" data-car-capacity="<?php echo (int)$row['capacity']; ?>">
                                <div class="modal-dialog">
                                    <div class="modal-content">
                                        <div class="modal-header">
                                            <h5 class="modal-title" id="bookingModalLabel<?php echo $row['id']; ?>">
                                                Book <?php echo htmlspecialchars($row['car_name']); ?>
                                            </h5>
                                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                        </div>
                                        <div class="modal-body">
                                            <form method="post" action="">
                                                <input type="hidden" name="car_id" value="<?php echo $row['id']; ?>">
                                                <!-- FIX: duration defaults to 12hrs; no $user_inputs dependency -->
                                                <input type="hidden" name="duration" value="12hrs">

                                                <!-- Hidden Rate Fields — now reading real DB values via COALESCE in query -->
                                                <input type="hidden" id="rate_inside_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_inside_zambales_12hrs']; ?>">
                                                <input type="hidden" id="rate_inside_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_inside_zambales_24hrs']; ?>">
                                                <input type="hidden" id="rate_outside_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_outside_zambales_12hrs']; ?>">
                                                <input type="hidden" id="rate_outside_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_outside_zambales_24hrs']; ?>">
                                                <input type="hidden" id="rate_baguio_12_<?php echo $row['id']; ?>" value="<?php echo $row['rate_baguio_12hrs']; ?>">
                                                <input type="hidden" id="rate_baguio_24_<?php echo $row['id']; ?>" value="<?php echo $row['rate_baguio_24hrs']; ?>">
                                                <input type="hidden" id="driver_price_12_<?php echo $row['id']; ?>" value="<?php echo $row['driver_price_12hrs']; ?>">
                                                <input type="hidden" id="driver_price_24_<?php echo $row['id']; ?>" value="<?php echo $row['driver_price_24hrs']; ?>">
                                                <input type="hidden" id="with_driver_<?php echo $row['id']; ?>" value="<?php echo $row['with_driver'] ? 'true' : 'false'; ?>">
                                                <input type="hidden" id="stakeholder_id_<?php echo $row['id']; ?>" value="<?php echo $row['stakeholder_id']; ?>">

                                                <!-- Full Name -->
                                                <div class="mb-3">
                                                    <label class="form-label">Full Name</label>
                                                    <input type="text" class="form-control" value="<?php echo htmlspecialchars($user_name); ?>" disabled>
                                                </div>

                                                <!-- Address -->
                                                <div class="mb-3">
                                                    <label for="address<?php echo $row['id']; ?>" class="form-label">Pickup Address <span class="text-danger">*</span></label>
                                                    <input type="text" class="form-control" id="address<?php echo $row['id']; ?>"
                                                        name="address" placeholder="Enter your pickup location" required>
                                                </div>

                                                <!-- Destination -->
                                                <div class="mb-3">
                                                    <label for="destination<?php echo $row['id']; ?>" class="form-label">Destination <span class="text-danger">*</span></label>
                                                    <select class="form-select" id="destination<?php echo $row['id']; ?>"
                                                        name="destination" required onchange="updateTotalPrice(<?php echo $row['id']; ?>)">
                                                        <option value="">Select Destination</option>
                                                        <?php foreach ($destinations as $dest): ?>
                                                            <option value="<?php echo htmlspecialchars($dest); ?>">
                                                                <?php echo htmlspecialchars($dest); ?>
                                                            </option>
                                                        <?php endforeach; ?>
                                                    </select>
                                                </div>

                                                <div class="mb-3">
                                                    <label for="trip_purpose<?php echo $row['id']; ?>" class="form-label">
                                                        Trip Purpose <span class="text-danger">*</span>
                                                    </label>
                                                    <input type="text"
                                                           class="form-control"
                                                           id="trip_purpose<?php echo $row['id']; ?>"
                                                           name="trip_purpose"
                                                           placeholder="e.g. Family Trip, Beach Trip, Business..."
                                                           required>
                                                </div>

                                                <!-- Number of Seaters -->
                                                <div class="mb-3">
                                                    <label for="number_of_seaters<?php echo $row['id']; ?>" class="form-label">
                                                        Number of Passengers <span class="text-danger">*</span>
                                                        <small class="text-muted">(Max: <?php echo $row['capacity']; ?> seats)</small>
                                                    </label>
                                                    <input type="number" class="form-control"
                                                        id="number_of_seaters<?php echo $row['id']; ?>"
                                                        name="number_of_seaters"
                                                        min="1"
                                                        max="<?php echo $row['capacity']; ?>"
                                                        placeholder="Enter number of passengers"
                                                        required>
                                                </div>

                                                <!-- Booking Status -->
                                                <div class="mb-3">
                                                    <label class="form-label">Booking Status</label>
                                                    <input type="text" class="form-control" value="Pending" disabled>
                                                </div>

                                                <!-- Payment Status -->
                                                <div class="mb-3">
                                                    <label class="form-label">Payment Status</label>
                                                    <input type="text" class="form-control" value="Unpaid" disabled>
                                                </div>

                                                <!-- Booking Date & Time -->
                                                <div class="mb-3">
                                                    <label for="booking_date<?php echo $row['id']; ?>" class="form-label">Booking Date & Time <span class="text-danger">*</span></label>
                                                    <input type="datetime-local" class="form-control"
                                                        id="booking_date<?php echo $row['id']; ?>"
                                                        name="booking_date"
                                                        required
                                                        min="<?php echo date('Y-m-d\TH:i'); ?>"
                                                        value="<?php echo date('Y-m-d\TH:i'); ?>"
                                                        onchange="updateTotalPrice(<?php echo $row['id']; ?>); validateBookingDate(<?php echo $row['id']; ?>)">
                                                    <small id="booking_date_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
                                                        Booking date must be in the future.
                                                    </small>
                                                </div>

                                                <!-- Return Date & Time -->
                                                <div class="mb-3">
                                                    <label for="return_date<?php echo $row['id']; ?>" class="form-label">Return Date & Time <span class="text-danger">*</span></label>
                                                    <input type="datetime-local" class="form-control"
                                                        id="return_date<?php echo $row['id']; ?>"
                                                        name="return_date"
                                                        required
                                                        min="<?php echo date('Y-m-d\TH:i'); ?>"
                                                        onchange="updateTotalPrice(<?php echo $row['id']; ?>); validateReturnDate(<?php echo $row['id']; ?>)">
                                                    <small id="return_date_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
                                                        Return date must be after booking date.
                                                    </small>
                                                    <small id="mk_rental_error_<?php echo $row['id']; ?>" class="text-danger" style="display: none;">
                                                        MK Car Rental requires minimum 48 hours.
                                                    </small>
                                                </div>

                                                <!-- Total Price -->
                                                <div class="mb-3">
                                                    <label class="form-label">Total Price</label>
                                                    <input type="text" class="form-control fw-bold text-primary"
                                                        id="total_price<?php echo $row['id']; ?>"
                                                        value="₱0.00" disabled>
                                                </div>

                                                <div class="mb-3">
                                                    <small class="text-muted">• Overstay by 1 hour = extra full rate charge</small><br>
                                                    <small class="text-muted">• No refund for early return</small>
                                                    <?php if ($row['stakeholder_id'] == 7): ?>
                                                        <br><small class="text-warning fw-bold">• MK Car Rental: Minimum 2 days (48 hours)</small>
                                                    <?php endif; ?>
                                                </div>

                                                <hr class="my-4">

                                                <!-- RENTAL AGREEMENT -->
                                                <div class="mb-3">
                                                    <h6 class="fw-bold text-primary mb-3">
                                                        <i class="bi bi-file-text-fill me-2"></i>Car Rental Contract Agreement
                                                    </h6>

                                                    <div class="border rounded p-3 mb-3" style="max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
                                                        <p class="small mb-2">
                                                            <strong>This Car Rental Contract Agreement</strong> ("Agreement") is entered into by and between the
                                                            Car Rental Provider (hereinafter referred to as the "Company") and the Customer/Renter
                                                            (hereinafter referred to as the "Customer"). By confirming the booking and using the rented vehicle,
                                                            the Customer agrees to all the terms and conditions stated herein.
                                                        </p>

                                                        <p class="small mb-2"><strong>1. AGREEMENT TO PAY RENTAL FEES</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer agrees to pay the total rental amount as displayed, quoted, or calculated by the Company
                                                            at the time of booking or vehicle release. This amount includes the agreed rental duration, vehicle type,
                                                            and other applicable charges disclosed prior to confirmation. The Customer acknowledges that failure to
                                                            pay the total agreed amount may result in penalties, additional charges, or legal action in accordance
                                                            with applicable laws.
                                                        </p>

                                                        <p class="small mb-2"><strong>2. ADDITIONAL CHARGES FOR EXCEEDED RENTAL DURATION</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer agrees that if the rented vehicle is returned beyond the agreed rental duration, additional
                                                            charges shall apply. These charges will be calculated based on the Company's prevailing rate (hourly or daily)
                                                            and shall be paid immediately upon vehicle return. Any delay without prior notice or approval from the Company
                                                            may be subject to extra penalties.
                                                        </p>

                                                        <p class="small mb-2"><strong>3. DAMAGE, LOSS, AND REPAIR LIABILITY</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer agrees to be fully responsible for any damage, loss, or theft of the rented vehicle during
                                                            the rental period, except for normal wear and tear. If the vehicle is returned with damages, the Customer
                                                            agrees to: (1) Pay the assessed damage repair cost, and/or (2) Cover replacement costs for missing or
                                                            damaged parts, and/or (3) Pay any applicable insurance excess or damage fee.
                                                        </p>

                                                        <p class="small mb-2"><strong>4. DRIVER'S LICENSE AND REQUIREMENTS</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer confirms and warrants that: (1) The designated driver holds a valid and non-expired driver's
                                                            license appropriate for the vehicle type rented; (2) All documents and requirements submitted to the Company
                                                            are true, accurate, and valid; (3) Only authorized and licensed drivers approved by the Company will operate
                                                            the rented vehicle.
                                                        </p>

                                                        <p class="small mb-2"><strong>5. SAFE AND RESPONSIBLE USE OF VEHICLE</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer agrees to operate the rented vehicle in a safe, careful, and lawful manner at all times.
                                                            The Customer shall: (1) Follow all traffic laws and regulations; (2) Avoid reckless, negligent, or dangerous
                                                            driving; (3) Not use the vehicle for illegal activities, racing, or prohibited purposes; (4) Not drive under
                                                            the influence of alcohol, drugs, or any impairing substances. Any violation resulting in fines, penalties,
                                                            or damage shall be the sole responsibility of the Customer.
                                                        </p>

                                                        <p class="small mb-2"><strong>6. ASSUMPTION OF RESPONSIBILITY</strong></p>
                                                        <p class="small mb-2">
                                                            The Customer assumes full responsibility for the vehicle from the time it is released by the Company until
                                                            it is properly returned and accepted by the Company.
                                                        </p>

                                                        <p class="small mb-0"><strong>7. ACCEPTANCE OF TERMS</strong></p>
                                                        <p class="small mb-0">
                                                            By signing this Agreement or confirming the rental booking, the Customer acknowledges that they have read,
                                                            understood, and agreed to all the terms and conditions stated herein.
                                                        </p>
                                                    </div>

                                                    <!-- Agreement Checkbox -->
                                                    <div class="form-check mb-3 p-3 border rounded" style="background-color: #fff3cd;">
                                                        <input type="checkbox" class="form-check-input"
                                                            id="terms_accepted_<?= $row['id'] ?>"
                                                            name="terms_accepted"
                                                            required
                                                            onchange="toggleSubmitButton(<?= $row['id'] ?>)">
                                                        <label class="form-check-label fw-bold" for="terms_accepted_<?= $row['id'] ?>">
                                                            <i class="bi bi-check-circle text-success me-1"></i>
                                                            I have read and agree to all the terms and conditions of this Car Rental Contract Agreement
                                                        </label>
                                                    </div>

                                                    <!-- Price Agreement Checkbox -->
                                                    <div class="form-check mb-3 p-3 border rounded" style="background-color: #d1ecf1;">
                                                        <input type="checkbox" class="form-check-input"
                                                            id="agree_price_<?= $row['id'] ?>"
                                                            required
                                                            onchange="toggleSubmitButton(<?= $row['id'] ?>)">
                                                        <label class="form-check-label" for="agree_price_<?= $row['id'] ?>">
                                                            I agree to pay <strong id="final_price_display_<?= $row['id'] ?>">₱0.00</strong> for this rental.<br>
                                                            <small class="text-muted">• No refund for early return • Overstay = full extra day charge</small>
                                                        </label>
                                                    </div>
                                                </div>

                                                <!-- Submit Button -->
                                                <button type="submit" name="book_car"
                                                    class="btn btn-primary w-100 py-2 fw-bold"
                                                    id="submit_btn_<?= $row['id'] ?>"
                                                    disabled>
                                                    <i class="bi bi-check-circle me-2"></i>Confirm Booking
                                                </button>

                                                <p class="text-center text-muted small mt-3 mb-0">
                                                    <i class="bi bi-shield-check me-1"></i>Your booking will be reviewed and confirmed by the car owner
                                                </p>
                                            </form>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="card car-card h-100 shadow-sm">
                                <?php
                                $images_query = $conn->query("SELECT image_path, is_primary FROM car_images WHERE car_id = {$row['id']} ORDER BY is_primary DESC, id");
                                $all_images = $images_query->fetch_all(MYSQLI_ASSOC);
                                $primary_img = $row['car_image'] ?? 'PLACEHOLDER.png';
                                ?>
                                <div class="position-relative">
                                    <img src="../stakeholders/uploads/cars/<?= htmlspecialchars($primary_img) ?>"
                                        class="car-img-top gallery-trigger"
                                        alt="<?= htmlspecialchars($row['car_name']) ?>"
                                        data-car-id="<?= $row['id'] ?>"
                                        data-images='<?= json_encode($all_images) ?>'>
                                    <?php if (count($all_images) > 1): ?>
                                        <div class="position-absolute bottom-0 start-0 m-3">
                                            <span class="badge bg-dark opacity-75">
                                                <i class="bi bi-images"></i> <?= count($all_images) ?> photos
                                            </span>
                                        </div>
                                    <?php endif; ?>
                                </div>
                                <div class="card-body d-flex flex-column">
                                    <!-- Car Name + Shop -->
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <h5 class="car-title mb-0"><?= htmlspecialchars($row['car_name']) ?></h5>
                                        <span class="badge bg-secondary"><?= htmlspecialchars($row['shop_name']) ?></span>
                                    </div>
                                    <!-- Price -->
                                    <div class="d-flex justify-content-end align-items-center mb-3">
                                        <span class="car-price">
                                            ₱<?= number_format($total_price, 2) ?>
                                            <small class="text-muted">/12hrs</small>
                                            <?php if ($row['with_driver'] && $row['driver_price'] > 0): ?>
                                                <br><small class="text-success">(Incl. ₱<?= number_format($row['driver_price'], 2) ?> driver fee)</small>
                                            <?php endif; ?>
                                        </span>
                                    </div>
                                    <!-- Total Bookings -->
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <small class="text-muted">
                                            <i class="bi bi-check-circle-fill text-success me-1"></i>
                                            Booked <strong><?= $row['booking_count'] ?></strong> <?= $row['booking_count'] == 1 ? 'time' : 'times' ?>
                                        </small>
                                    </div>
                                    <!-- Enhanced Features List -->
                                    <ul class="car-features list-unstyled mb-4">
                                        <li><i class="bi bi-people text-primary me-2"></i> <?= $row['capacity'] ?> Seats (<?= $row['seating_capacity'] ?? 'N/A' ?>)</li>
                                        <li><i class="bi bi-paint-bucket text-info me-2"></i> <?= $row['color'] ?? 'N/A' ?></li>
                                        <li><i class="bi bi-gear text-secondary me-2"></i> <?= $row['transmission'] ?></li>
                                        <li><i class="bi bi-fuel-pump text-warning me-2"></i> <?= $row['fuel_type'] ?></li>
                                        <li><i class="bi bi-car-front text-success me-2"></i> <?= $row['car_type'] ?></li>

                                        <?php if (!empty($row['trip_purpose'])):
                                            $purposes = json_decode($row['trip_purpose'], true);
                                            if (is_array($purposes) && !empty($purposes)):
                                        ?>
                                                <li><i class="bi bi-signpost-2 text-purple me-2"></i>
                                                    <strong>Best for:</strong> <?= htmlspecialchars(implode(', ', $purposes)) ?>
                                                </li>
                                        <?php endif;
                                        endif; ?>

                                        <?php if ($row['terrain_compatibility'] ?? false): ?>
                                            <li><i class="bi bi-map-fill text-success me-2"></i> <?= $row['terrain_compatibility'] ?></li>
                                        <?php endif; ?>

                                        <?php if ($row['ground_clearance'] ?? false): ?>
                                            <li><i class="bi bi-arrow-up-circle text-info me-2"></i> <?= str_replace('Ground Clearance', '', $row['ground_clearance']) ?></li>
                                        <?php endif; ?>

                                        <?php if ($row['cargo_space'] ?? false): ?>
                                            <li><i class="bi bi-box-seam text-primary me-2"></i> <?= $row['cargo_space'] ?></li>
                                        <?php endif; ?>

                                        <?php if ($row['compact'] ?? false && $row['compact'] === 'Yes'): ?>
                                            <li><i class="bi bi-arrows-collapse text-indigo me-2"></i> Compact & Easy to Park</li>
                                        <?php endif; ?>

                                        <?php if ($row['special_needs_friendly'] == 'Y'): ?>
                                            <li><i class="bi bi-heart-fill text-danger me-2"></i> PWD & Senior Friendly</li>
                                        <?php endif; ?>
                                        <?php if ($row['child_seat'] == 'Y'): ?>
                                            <li><i class="bi bi-baby-carriage text-pink me-2"></i> Child Seat Available</li>
                                        <?php endif; ?>
                                        <?php if ($row['wide_leg_room'] == 'Y'): ?>
                                            <li><i class="bi bi-arrows-expand text-indigo me-2"></i> Wide Leg Room</li>
                                        <?php endif; ?>
                                        <?php if ($row['aircon'] == 'Y'): ?>
                                            <li><i class="bi bi-fan text-info me-2"></i> Air Conditioned</li>
                                        <?php endif; ?>
                                        <?php if ($row['budget_friendly'] == 'Y'): ?>
                                            <li><i class="bi bi-wallet text-success me-2"></i> Budget Friendly</li>
                                        <?php endif; ?>
                                        <li><i class="bi bi-person text-dark me-2"></i> <?= $row['with_driver'] ? 'With Driver' : 'Self-Drive' ?></li>
                                    </ul>
                                    <!-- Action Buttons -->
                                    <div class="d-grid mt-auto gap-2">
                                        <button class="btn btn-primary btn-lg"
                                            data-bs-toggle="modal"
                                            data-bs-target="#bookingModal<?= $row['id'] ?>"
                                            data-car-capacity="<?= (int)$row['capacity'] ?>">
                                            Book Now
                                        </button>
                                        <button type="button" class="btn btn-outline-primary view-reviews-btn"
                                            data-bs-toggle="modal" data-bs-target="#reviewsModal"
                                            data-car-id="<?= $row['id'] ?>" data-car-name="<?= htmlspecialchars($row['car_name']) ?>">
                                            View Reviews
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    <?php
                    endwhile;
                else:
                    ?>
                    <div class="col-12">
                        <?php if ($search_performed): ?>
                            <div class="alert alert-warning text-center">
                                <i class="bi bi-search fs-1 mb-3"></i><br>
                                <h5>No cars found for "<?= htmlspecialchars($search_query) ?>"</h5>
                                <p class="mb-3">Try searching with different keywords like:</p>
                                <div class="row g-2 justify-content-center">
                                    <div class="col-auto"><a href="?search=Toyota" class="btn btn-outline-primary btn-sm">Toyota</a></div>
                                    <div class="col-auto"><a href="?search=SUV" class="btn btn-outline-primary btn-sm">SUV</a></div>
                                    <div class="col-auto"><a href="?search=Automatic" class="btn btn-outline-primary btn-sm">Automatic</a></div>
                                    <div class="col-auto"><a href="?search=Budget" class="btn btn-outline-primary btn-sm">Budget</a></div>
                                </div>
                                <hr>
                                <a href="?" class="btn btn-primary">View All Cars</a>
                            </div>
                        <?php else: ?>
                            <div class="alert alert-info text-center">
                                <i class="bi bi-car-front fs-1"></i><br>
                                No cars available at the moment.
                            </div>
                        <?php endif; ?>
                    </div>
                <?php endif; ?>
            </div>

            <!-- Pagination -->
            <?php if ($total_pages > 1): ?>
                <nav class="mt-5" aria-label="Page navigation">
                    <ul class="pagination justify-content-center">
                        <li class="page-item <?= $current_page <= 1 ? 'disabled' : '' ?>">
                            <a class="page-link"
                                href="?<?= $search_performed ? 'search=' . urlencode($search_query) . '&search_page=' . ($current_page - 1) : 'page=' . ($current_page - 1) ?>"
                                aria-label="Previous">
                                <span aria-hidden="true">Previous</span>
                            </a>
                        </li>
                        <?php
                        $start = max(1, $current_page - 2);
                        $end = min($total_pages, $current_page + 2);
                        if ($start > 1):
                        ?>
                            <li class="page-item">
                                <a class="page-link"
                                    href="?<?= $search_performed ? 'search=' . urlencode($search_query) . '&search_page=1' : 'page=1' ?>">1</a>
                            </li>
                            <?php if ($start > 2): ?><li class="page-item disabled"><span class="page-link">...</span></li><?php endif; ?>
                        <?php endif; ?>
                        <?php for ($i = $start; $i <= $end; $i++): ?>
                            <li class="page-item <?= $i == $current_page ? 'active' : '' ?>">
                                <a class="page-link"
                                    href="?<?= $search_performed ? 'search=' . urlencode($search_query) . '&search_page=' . $i : 'page=' . $i ?>"><?= $i ?></a>
                            </li>
                        <?php endfor; ?>
                        <?php if ($end < $total_pages): ?>
                            <?php if ($end < $total_pages - 1): ?><li class="page-item disabled"><span class="page-link">...</span></li><?php endif; ?>
                            <li class="page-item">
                                <a class="page-link"
                                    href="?<?= $search_performed ? 'search=' . urlencode($search_query) . '&search_page=' . $total_pages : 'page=' . $total_pages ?>"><?= $total_pages ?></a>
                            </li>
                        <?php endif; ?>
                        <li class="page-item <?= $current_page >= $total_pages ? 'disabled' : '' ?>">
                            <a class="page-link"
                                href="?<?= $search_performed ? 'search=' . urlencode($search_query) . '&search_page=' . ($current_page + 1) : 'page=' . ($current_page + 1) ?>"
                                aria-label="Next">
                                <span aria-hidden="true">Next</span>
                            </a>
                        </li>
                    </ul>
                </nav>
            <?php endif; ?>
        </div>
    </section>
    <!-- Gallery Modal (One for all cars) -->
    <div class="modal fade" id="galleryModal" tabindex="-1">
        <div class="modal-dialog modal-lg modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Car Gallery</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body text-center">
                    <div id="galleryCarousel" class="carousel slide">
                        <div class="carousel-inner" id="galleryImages"></div>
                        <button class="carousel-control-prev" type="button" data-bs-target="#galleryCarousel" data-bs-slide="prev">
                            <span class="carousel-control-prev-icon"></span>
                        </button>
                        <button class="carousel-control-next" type="button" data-bs-target="#galleryCarousel" data-bs-slide="next">
                            <span class="carousel-control-next-icon"></span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <!-- Reviews Modal -->
    <div class="modal fade" id="reviewsModal" tabindex="-1" aria-labelledby="reviewsModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="reviewsModalLabel">Reviews for <span id="modalCarName"></span></h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div id="reviewsContent">
                        <p>Loading reviews...</p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        $(document).ready(function() {
            $('.view-reviews-btn').on('click', function() {
                const carId = $(this).data('car-id');
                const carName = $(this).data('car-name');
                $('#modalCarName').text(carName);

                $.ajax({
                    url: 'get_reviews.php',
                    method: 'POST',
                    data: {
                        car_id: carId
                    },
                    success: function(response) {
                        $('#reviewsContent').html(response);
                    },
                    error: function() {
                        $('#reviewsContent').html('<p class="text-danger">Error loading reviews. Please try again later.</p>');
                    }
                });
            });
        });
    </script>

    <script>
        function toggleSubmitButton(carId) {
            const termsChecked = document.getElementById(`terms_accepted_${carId}`)?.checked || false;
            const priceChecked = document.getElementById(`agree_price_${carId}`)?.checked || false;
            const submitBtn = document.getElementById(`submit_btn_${carId}`);
            const currentPrice = document.getElementById(`total_price${carId}`).value;

            const hasValidPrice = currentPrice !== '₱0.00' && !currentPrice.includes('Invalid');
            const bookingDateError = document.getElementById(`booking_date_error_${carId}`)?.style.display !== 'block';
            const returnDateError = document.getElementById(`return_date_error_${carId}`)?.style.display !== 'block';
            const mkError = document.getElementById(`mk_rental_error_${carId}`)?.style.display !== 'block';

            submitBtn.disabled = !(termsChecked && priceChecked && hasValidPrice && bookingDateError && returnDateError && mkError);
        }


        // Gallery Modal Trigger
        document.addEventListener('click', function(e) {
            if (e.target.matches('.gallery-trigger')) {
                const carId = e.target.dataset.carId;
                const images = JSON.parse(e.target.dataset.images);

                const carouselInner = document.getElementById('galleryImages');
                carouselInner.innerHTML = '';

                images.forEach((img, index) => {
                    const item = document.createElement('div');
                    item.className = 'carousel-item' + (index === 0 ? ' active' : '');
                    item.innerHTML = `
                <img src="../stakeholders/uploads/cars/${img.image_path}" 
                     class="d-block w-100" 
                     style="max-height: 70vh; object-fit: contain; background: #000;"
                     alt="Car image ${index + 1}">
            `;
                    carouselInner.appendChild(item);
                });

                new bootstrap.Modal(document.getElementById('galleryModal')).show();
            }
        });


        function updateTotalPrice(carId) {

            const destination = document.getElementById(`destination${carId}`).value;
            const bookingDate = document.getElementById(`booking_date${carId}`).value;
            const returnDate = document.getElementById(`return_date${carId}`).value;
            if (!destination || !bookingDate || !returnDate) {
                document.getElementById(`total_price${carId}`).value = '₱0.00';
                updateAgreementDisplay(carId, 0);
                return;
            }
            const durationInput = document.querySelector(`#bookingModal${carId} input[name="duration"]`);
            const duration = durationInput ? durationInput.value : '12hrs';
            const bookingTime = new Date(bookingDate);
            const returnTime = new Date(returnDate);
            if (returnTime <= bookingTime) {
                document.getElementById(`total_price${carId}`).value = 'Invalid date range';
                updateAgreementDisplay(carId, 0);
                return;
            }
            const rates = {
                inside_12: parseFloat(document.getElementById(`rate_inside_12_${carId}`).value) || 3000,
                inside_24: parseFloat(document.getElementById(`rate_inside_24_${carId}`).value) || 5000,
                outside_12: parseFloat(document.getElementById(`rate_outside_12_${carId}`).value) || 3500,
                outside_24: parseFloat(document.getElementById(`rate_outside_24_${carId}`).value) || 6000,
                baguio_12: parseFloat(document.getElementById(`rate_baguio_12_${carId}`).value) || 4000,
                baguio_24: parseFloat(document.getElementById(`rate_baguio_24_${carId}`).value) || 7000,
                driver_12: parseFloat(document.getElementById(`driver_price_12_${carId}`).value) || 500,
                driver_24: parseFloat(document.getElementById(`driver_price_24_${carId}`).value) || 800,
                withDriver: document.getElementById(`with_driver_${carId}`).value === 'true'
            };
            let baseRate = 0;
            if (destination === 'Baguio') baseRate = duration === '12hrs' ? rates.baguio_12 : rates.baguio_24;
            else if (destination === 'Inside Zambales') baseRate = duration === '12hrs' ? rates.inside_12 : rates.inside_24;
            else baseRate = duration === '12hrs' ? rates.outside_12 : rates.outside_24;
            const driverPrice = duration === '12hrs' ? rates.driver_12 : rates.driver_24;
            const hours = (returnTime - bookingTime) / (1000 * 60 * 60);
            const durationHours = duration === '12hrs' ? 12 : 24;
            const multiplier = Math.ceil(hours / durationHours);
            const total = (baseRate * multiplier) + (rates.withDriver ? driverPrice * multiplier : 0);

            const formattedPrice = '₱' + total.toLocaleString('en-PH', {
                minimumFractionDigits: 2
            });
            document.getElementById(`total_price${carId}`).value = formattedPrice;

            // Sync to agreement checkbox text
            updateAgreementDisplay(carId, total);
        }

        function updateAgreementDisplay(carId, totalPrice) {
            const display = document.getElementById(`final_price_display_${carId}`);
            const checkbox = document.getElementById(`agree_price_${carId}`);

            if (!display || !checkbox) return;

            if (totalPrice <= 0) {
                display.textContent = '₱0.00';
                checkbox.checked = false;
            } else {
                const formatted = '₱' + totalPrice.toLocaleString('en-PH', {
                    minimumFractionDigits: 2
                });
                display.textContent = formatted;
            }

            // Re-validate submit button state
            toggleSubmitButton(carId);
        }

        // Update existing checkbox change handlers
        document.addEventListener('change', function(e) {
            if (e.target.id.startsWith('agree_price_') || e.target.id.startsWith('terms_accepted_')) {
                const carId = e.target.id.replace('agree_price_', '').replace('terms_accepted_', '');
                toggleSubmitButton(carId);
            }
        });
        // Also run on modal show to initialize agreement state
        document.querySelectorAll('[data-bs-target^="\\#bookingModal"]').forEach(btn => {
            btn.addEventListener('click', function() {
                const modalId = this.getAttribute('data-bs-target').substring(1);
                const carId = modalId.replace('bookingModal', '');
                setTimeout(() => {
                    const now = new Date();
                    const dateStr = now.toISOString().slice(0, 16);
                    document.getElementById(`booking_date${carId}`).value = dateStr;
                    updateTotalPrice(carId); // This now also updates agreement text
                }, 150);
            });
        });

        function validateBookingDate(carId) {
            const bookingDate = document.getElementById(`booking_date${carId}`).value;
            if (!bookingDate) return;

            const now = new Date();
            const bookingTime = new Date(bookingDate);
            const error = document.getElementById(`booking_date_error_${carId}`);
            const submitBtn = document.getElementById(`submit_btn_${carId}`);

            if (bookingTime < now) {
                error.style.display = 'block';
                submitBtn.disabled = true;
            } else {
                error.style.display = 'none';
                // Re-validate return date
                validateReturnDate(carId);
            }

            // Update return date minimum to be after booking date
            const returnDateInput = document.getElementById(`return_date${carId}`);
            const minReturn = new Date(bookingTime.getTime() + 60000).toISOString().slice(0, 16);
            returnDateInput.min = minReturn;
        }

        function validateReturnDate(carId) {
            const bookingDate = document.getElementById(`booking_date${carId}`).value;
            const returnDate = document.getElementById(`return_date${carId}`).value;

            if (!bookingDate || !returnDate) return;

            const bookingTime = new Date(bookingDate);
            const returnTime = new Date(returnDate);
            const hours = (returnTime - bookingTime) / (1000 * 60 * 60);

            const error = document.getElementById(`return_date_error_${carId}`);
            const mkError = document.getElementById(`mk_rental_error_${carId}`);
            const submitBtn = document.getElementById(`submit_btn_${carId}`);
            const stakeholderId = document.getElementById(`stakeholder_id_${carId}`).value;

            let hasError = false;

            // Check if return date is after booking date
            if (returnTime <= bookingTime) {
                error.style.display = 'block';
                hasError = true;
            } else {
                error.style.display = 'none';
            }

            // Check MK Car Rental 48-hour minimum
            if (stakeholderId == 7 && hours < 48) {
                mkError.style.display = 'block';
                hasError = true;
            } else {
                mkError.style.display = 'none';
            }

            submitBtn.disabled = hasError;
        }

        // Initialize total price on modal open
        document.querySelectorAll('[data-bs-target^="\\#bookingModal"]').forEach(btn => {
            btn.addEventListener('click', function() {
                const modalId = this.getAttribute('data-bs-target').substring(1);
                const carId = modalId.replace('bookingModal', '');
                setTimeout(() => {
                    // Set default booking date to now
                    const now = new Date();
                    const dateStr = now.toISOString().slice(0, 16);
                    document.getElementById(`booking_date${carId}`).value = dateStr;
                    updateTotalPrice(carId);
                }, 100);
            });
        });
    </script>
    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="row g-4">
                <div class="col-lg-4 mb-4 mb-lg-0">
                    <h3 class="footer-heading">CarZam</h3>
                    <p class="mb-4">The intelligent car rental recommendation system that helps you find the perfect vehicle for your needs at the best price.</p>
                    <div class="social-icons mb-4">
                        <a href="#" class="social-icon"><i class="bi bi-facebook"></i></a>
                        <a href="#" class="social-icon"><i class="bi bi-twitter"></i></a>
                        <a href="#" class="social-icon"><i class="bi bi-instagram"></i></a>
                        <a href="#" class="social-icon"><i class="bi bi-linkedin"></i></a>
                    </div>
                </div>
                <div class="col-sm-6 col-lg-2 mb-4 mb-lg-0">
                    <h5 class="footer-heading">Quick Links</h5>
                    <a href="#" class="footer-link">Home</a>
                    <a href="#" class="footer-link">Cars</a>
                    <a href="#" class="footer-link">How It Works</a>
                    <a href="#" class="footer-link">About Us</a>
                    <a href="#" class="footer-link">Contact</a>
                </div>
                <div class="col-sm-6 col-lg-3 mb-4 mb-lg-0">
                    <h5 class="footer-heading">Car Types</h5>
                    <a href="#" class="footer-link">Economy</a>
                    <a href="#" class="footer-link">Compact</a>
                    <a href="#" class="footer-link">Mid-size</a>
                    <a href="#" class="footer-link">SUV</a>
                    <a href="#" class="footer-link">Luxury</a>
                </div>
                <div class="col-lg-3">
                    <h5 class="footer-heading">Contact Us</h5>
                    <p class="mb-1"><i class="bi bi-geo-alt me-2"></i> 123 Car Street, Auto City</p>
                    <p class="mb-1"><i class="bi bi-telephone me-2"></i> (123) 456-7890</p>
                    <p class="mb-3"><i class="bi bi-envelope me-2"></i> info@carzam.com</p>
                    <h5 class="footer-heading mt-4">Newsletter</h5>
                    <div class="input-group">
                        <input type="email" class="form-control" placeholder="Your email">
                        <button class="btn btn-primary" type="button">Subscribe</button>
                    </div>
                </div>
            </div>
            <hr class="mt-5 mb-4" style="border-color: var(--light-blue);">
            <div class="text-center">
                <p class="mb-0">&copy; 2025 CarZam. All rights reserved.</p>
            </div>
        </div>
    </footer>


</body>

</html>