<?php
// Set timezone to Asia/Manila (GMT+8)
date_default_timezone_set('Asia/Manila');
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
// 2. BOOKING LOGIC (100% preserved)
// ---------------------------------------------------------------------
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['book_car'])) {
    $car_id      = filter_var($_POST['car_id'], FILTER_VALIDATE_INT);
    $duration    = $_POST['duration'] ?? '12hrs';
    $destination = $conn->real_escape_string($_POST['destination']);
    $address     = $conn->real_escape_string($_POST['address']);
    $return_date = $conn->real_escape_string($_POST['return_date']);
    $booking_date = date('Y-m-d H:i:s');
    $booking_datetime = new DateTime($booking_date);
    $return_datetime  = new DateTime($return_date);

    if ($return_datetime <= $booking_datetime) {
        $_SESSION['booking_error'] = "Return date must be after booking date.";
        header("Location: " . $_SERVER['PHP_SELF']);
        exit;
    }

    $interval = $booking_datetime->diff($return_datetime);
    $hours    = $interval->days * 24 + $interval->h;

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
    $rate_data   = $rate_result && $rate_result->num_rows > 0 ? $rate_result->fetch_assoc() : null;

    if (!$rate_data || $rate_data['base_rate'] === null) {
        $avg_query = "SELECT AVG($rate_field) AS avg_rate FROM car_rates WHERE $rate_field IS NOT NULL";
        $avg_result = $conn->query($avg_query);
        $base_rate = $avg_result->fetch_assoc()['avg_rate'] ?? 0;
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

    if ($car_id && $total_price && $user_id && $address && $destination && $return_date) {
        $avail_query = "SELECT status FROM car_availability WHERE car_id = $car_id";
        $avail_result = $conn->query($avail_query);
        if ($avail_result && $avail_result->num_rows > 0 && $avail_result->fetch_assoc()['status'] === 'available') {
            $booking_query = "INSERT INTO bookings (user_id, car_id, booking_status, payment_status, booking_date, return_date, total_price, duration, destination, address, created_at)
                            VALUES ($user_id, $car_id, 'pending', 'unpaid', '$booking_date', '$return_date', $total_price, '$duration', '$destination', '$address', CURRENT_TIMESTAMP)";
            if ($conn->query($booking_query)) {
                $booking_id = $conn->insert_id;
                $stakeholder_query = "SELECT stakeholder_id FROM car_inventory WHERE id = $car_id";
                $stakeholder_result = $conn->query($stakeholder_query);
                $stakeholder_id = $stakeholder_result && $stakeholder_result->num_rows > 0 ? $stakeholder_result->fetch_assoc()['stakeholder_id'] : null;
                if ($stakeholder_id) {
                    $history_query = "INSERT INTO booking_history (car_id, user_id, stakeholder_id, start_date, end_date, booking_status, payment_status, total_amount)
                                    VALUES ($car_id, $user_id, $stakeholder_id, '$booking_date', '$return_date', 'pending', 'unpaid', $total_price)";
                    $conn->query($history_query);
                }
                $update_avail_query = "UPDATE car_availability SET status = 'dispatched', updated_at = CURRENT_TIMESTAMP WHERE car_id = $car_id";
                $conn->query($update_avail_query);
                $_SESSION['booking_success'] = "Booking successfully preserved! Awaiting confirmation.";
            } else {
                $_SESSION['booking_error'] = "Failed to create booking. Please try again.";
            }
        } else {
            $_SESSION['booking_error'] = "Selected car is not available for booking.";
        }
    } else {
        $_SESSION['booking_error'] = "Invalid booking details.";
    }
    header("Location: " . $_SERVER['PHP_SELF']);
    exit;
}

// ---------------------------------------------------------------------
// 3. DEFAULT CARS + PAGINATION
// ---------------------------------------------------------------------
$results_per_page = 6;
$current_page = max(1, (int)($_GET['page'] ?? 1));
$query = "SELECT
            ci.*,
            sa.shop_name, sa.location,
            cr.rate_inside_zambales_12hrs,
            cr.rate_inside_zambales_24hrs,
            cr.rate_outside_zambales_12hrs,
            cr.rate_outside_zambales_24hrs,
            cr.rate_baguio_12hrs,
            cr.rate_baguio_24hrs,
            cr.with_driver,
            COALESCE(cr.driver_price_12hrs, 0) AS driver_price_12hrs,
            COALESCE(cr.driver_price_24hrs, 0) AS driver_price_24hrs,
            (SELECT COUNT(*) FROM booking_history bh 
             WHERE bh.car_id = ci.id 
             AND bh.booking_status IN ('confirmed','completed')) AS booking_count
          FROM car_inventory ci
          JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
          LEFT JOIN car_rates cr ON ci.id = cr.car_id
          JOIN car_availability ca ON ci.id = ca.car_id
          WHERE ca.status = 'available'
          ORDER BY booking_count DESC
          LIMIT " . (($current_page - 1) * $results_per_page) . ", $results_per_page";

$result = $conn->query($query);
$count_query = "SELECT COUNT(*) as total FROM car_inventory ci 
                JOIN car_availability ca ON ci.id = ca.car_id 
                WHERE ca.status = 'available'";
$count_result = $conn->query($count_query);
$total_records = $count_result->fetch_assoc()['total'];
$total_pages = ceil($total_records / $results_per_page);
?>
 

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
                            <li><a class="dropdown-item" href="reviews.php"><i class="bi bi-star"></i> Reviews</a></li>
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
    <!-- Search Section -->
    <!-- Search Section -->
    <section class="search-container py-5 bg-light">
        <div class="container">
            <h2 class="section-title mb-4">Find Your Perfect Rental Car</h2>
            <p class="lead mb-4">Get personalized car recommendations based on your preferences.</p>
            <div class="card shadow mt-2">
                <div class="card-body">
                    <form id="findCarsForm" class="mt-4">
                        <div class="row g-3">
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label">Car Type</label>
                                <select class="form-select" name="car_type" id="car_type" required>
                                    <option value="">Select</option>
                                    <option value="Sedan">Sedan</option>
                                    <option value="SUV">SUV</option>
                                    <option value="MPV">MPV</option>
                                    <option value="Van">Van</option>
                                    <option value="Mini Truck">Mini Truck</option>
                                </select>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label">Capacity</label>
                                <select class="form-select" name="capacity" id="capacity" required>
                                    <option value="">Auto</option>
                                </select>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label">Duration</label>
                                <select class="form-select" name="duration" required>
                                    <option value="">Select</option>
                                    <option value="12 Hours">12 Hours</option>
                                    <option value="24 Hours">24 Hours</option>
                                    <option value="Weekly">Weekly</option>
                                    <option value="Monthly">Monthly</option>
                                    <option value="Yearly">Yearly</option>
                                </select>
                            </div>
                            <div class="col-md-6 col-lg-2">
                                <label class="form-label">Budget (₱)</label>
                                <input type="number" class="form-control" name="budget" min="1000" step="500" required>
                            </div>
                            <div class="col-md-12 col-lg-4">
                                <label class="form-label">Additional Inquiries</label>
                                <textarea class="form-control" name="inquiry" rows="1" placeholder="e.g. diesel, child seat, matipid, etc."></textarea>
                            </div>
                            <div class="col-12 text-center mt-3">
                                <button type="submit" class="btn btn-primary btn-lg px-5">Find Cars</button>
                                <a href="MainSearch.php" class="btn btn-success btn-lg px-5 ms-2">Smart Search</a>
                                <button type="button" id="resetFilters" class="btn btn-outline-secondary ms-2">Reset Filters</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </section>

    <section class="car-section py-5">
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

            <h2 class="section-title">Available Cars</h2>
            <div class="row g-4" id="carContainer">
                <?php if ($result && $result->num_rows > 0):
                    while ($row = $result->fetch_assoc()):
                        // Use real base rate (12hrs inside Zambales) — 0 if null
                        $base_rate = $row['rate_inside_zambales_12hrs'] ?? 0;
                        $driver_fee = $row['with_driver'] ? ($row['driver_price_12hrs'] ?? 0) : 0;
                        $total_price = $base_rate + $driver_fee;
                ?>
                        <!-- ==== MODAL ==== -->
                        <div class="modal fade" id="bookingModal<?= $row['id'] ?>" tabindex="-1" aria-labelledby="bookingModalLabel<?= $row['id'] ?>" aria-hidden="true">
                            <div class="modal-dialog">
                                <div class="modal-content">
                                    <div class="modal-header">
                                        <h5 class="modal-title" id="bookingModalLabel<?= $row['id'] ?>">Book <?= htmlspecialchars($row['car_name']) ?></h5>
                                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                    </div>
                                    <div class="modal-body">
                                        <form method="post" action="">
                                            <input type="hidden" name="car_id" value="<?= $row['id'] ?>">
                                            <input type="hidden" name="duration" value="12hrs">

                                            <!-- REAL RATES FROM DB (NO FALLBACK) -->
                                            <input type="hidden" id="rate_inside_12_<?= $row['id'] ?>" value="<?= $row['rate_inside_zambales_12hrs'] ?? 0 ?>">
                                            <input type="hidden" id="rate_inside_24_<?= $row['id'] ?>" value="<?= $row['rate_inside_zambales_24hrs'] ?? 0 ?>">
                                            <input type="hidden" id="rate_outside_12_<?= $row['id'] ?>" value="<?= $row['rate_outside_zambales_12hrs'] ?? 0 ?>">
                                            <input type="hidden" id="rate_outside_24_<?= $row['id'] ?>" value="<?= $row['rate_outside_zambales_24hrs'] ?? 0 ?>">
                                            <input type="hidden" id="rate_baguio_12_<?= $row['id'] ?>" value="<?= $row['rate_baguio_12hrs'] ?? 0 ?>">
                                            <input type="hidden" id="rate_baguio_24_<?= $row['id'] ?>" value="<?= $row['rate_baguio_24hrs'] ?? 0 ?>">
                                            <input type="hidden" id="driver_price_12_<?= $row['id'] ?>" value="<?= $row['driver_price_12hrs'] ?? 0 ?>">
                                            <input type="hidden" id="driver_price_24_<?= $row['id'] ?>" value="<?= $row['driver_price_24hrs'] ?? 0 ?>">
                                            <input type="hidden" id="with_driver_<?= $row['id'] ?>" value="<?= $row['with_driver'] ? 'true' : 'false' ?>">
                                            <input type="hidden" id="stakeholder_id_<?= $row['id'] ?>" value="<?= $row['stakeholder_id'] ?>">

                                            <div class="mb-3"><label class="form-label">Full Name</label>
                                                <input type="text" class="form-control" value="<?= htmlspecialchars($user_name) ?>" disabled>
                                            </div>
                                            <div class="mb-3"><label for="address<?= $row['id'] ?>" class="form-label">Address</label>
                                                <input type="text" class="form-control" id="address<?= $row['id'] ?>" name="address" required>
                                            </div>
                                            <div class="mb-3"><label for="destination<?= $row['id'] ?>" class="form-label">Destination</label>
                                                <select class="form-select" id="destination<?= $row['id'] ?>" name="destination" required onchange="updateTotalPrice(<?= $row['id'] ?>)">
                                                    <option value="">Select Destination</option>
                                                    <?php foreach ($destinations as $dest): ?>
                                                        <option value="<?= htmlspecialchars($dest) ?>"><?= htmlspecialchars($dest) ?></option>
                                                    <?php endforeach; ?>
                                                </select>
                                            </div>
                                            <div class="mb-3"><label class="form-label">Booking Status</label>
                                                <input type="text" class="form-control" value="Pending" disabled>
                                            </div>
                                            <div class="mb-3"><label class="form-label">Payment Status</label>
                                                <input type="text" class="form-control" value="Unpaid" disabled>
                                            </div>
                                            <div class="mb-3"><label for="booking_date<?= $row['id'] ?>" class="form-label">Booking Date & Time</label>
                                                <input type="datetime-local" class="form-control" id="booking_date<?= $row['id'] ?>" name="booking_date" required min="<?= date('Y-m-d\TH:i') ?>" value="<?= date('Y-m-d\TH:i') ?>" onchange="updateTotalPrice(<?= $row['id'] ?>); validateBookingDate(<?= $row['id'] ?>)">
                                                <small id="booking_date_error_<?= $row['id'] ?>" class="text-danger" style="display:none;">Booking date must be in the future.</small>
                                            </div>
                                            <div class="mb-3"><label for="return_date<?= $row['id'] ?>" class="form-label">Return Date & Time</label>
                                                <input type="datetime-local" class="form-control" id="return_date<?= $row['id'] ?>" name="return_date" required min="<?= date('Y-m-d\TH:i') ?>" onchange="updateTotalPrice(<?= $row['id'] ?>); validateReturnDate(<?= $row['id'] ?>)">
                                                <small id="return_date_error_<?= $row['id'] ?>" class="text-danger" style="display:none;">Return date must be after booking date.</small>
                                                <small id="mk_rental_error_<?= $row['id'] ?>" class="text-danger" style="display:none;">MK Car Rental requires minimum 48 hours.</small>
                                            </div>
                                            <div class="mb-3"><label class="form-label">Total Price</label>
                                                <input type="text" class="form-control" id="total_price<?= $row['id'] ?>" value="₱0.00" disabled>
                                            </div>
                                            <div class="mb-3">
                                                <small class="text-muted">Overstay by 1 hour = extra full rate.</small>
                                                <?php if ($row['stakeholder_id'] == 7): ?>
                                                    <br><small class="text-muted">MK Car Rental: Minimum 2 days (48 hours).</small>
                                                <?php endif; ?>
                                            </div>
                                            <button type="submit" name="book_car" class="btn btn-primary w-100" id="submit_btn_<?= $row['id'] ?>">Confirm Booking</button>
                                        </form>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- ==== CAR CARD ==== -->
                        <div class="col-md-6 col-lg-4">
                            <div class="card car-card h-100 shadow-sm">
                                <div class="position-relative">
                                    <img src="../stakeholders/uploads/cars/<?= htmlspecialchars($row['car_image'] ?? 'PLACEHOLDER.png') ?>"
                                        class="car-img-top" alt="<?= htmlspecialchars($row['car_name']) ?>">
                                </div>
                                <div class="card-body d-flex flex-column">
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <h5 class="car-title mb-0"><?= htmlspecialchars($row['car_name']) ?></h5>
                                        <span class="badge bg-secondary"><?= htmlspecialchars($row['shop_name']) ?></span>
                                    </div>
                                    <div class="d-flex justify-content-end align-items-center mb-3">
                                        <span class="car-price">
                                            ₱<?= number_format($total_price, 2) ?>
                                            <small class="text-muted">/12hrs</small>
                                            <?php if ($row['with_driver'] && $driver_fee > 0): ?>
                                                <br><small class="text-success">(Incl. ₱<?= number_format($driver_fee, 2) ?> driver fee)</small>
                                            <?php endif; ?>
                                        </span>
                                    </div>
                                    <div class="d-flex justify-content-between align-items-center mb-3">
                                        <small class="text-muted">
                                            Booked <strong><?= $row['booking_count'] ?></strong> <?= $row['booking_count'] == 1 ? 'time' : 'times' ?>
                                        </small>
                                    </div>
                                    <ul class="car-features list-unstyled mb-4">
                                        <li><i class="bi bi-people-fill text-primary me-2"></i> <?= $row['capacity'] ?> Seats</li>
                                        <li><i class="bi bi-droplet-fill text-info me-2"></i> <?= $row['color'] ?? 'N/A' ?></li>
                                        <li><i class="bi bi-gear-fill text-secondary me-2"></i> <?= $row['transmission'] ?></li>
                                        <li><i class="bi bi-fuel-pump-fill text-success me-2"></i> <?= $row['fuel_type'] ?></li>
                                        <li><i class="bi bi-car-front-fill text-indigo me-2"></i> <?= $row['car_type'] ?></li>
                                        <?php if ($row['special_needs_friendly'] == 'Y'): ?>
                                            <li><i class="bi bi-universal-access-circle-fill text-warning me-2"></i> PWD & Senior Friendly</li>
                                        <?php endif; ?>
                                        <?php if ($row['child_seat'] == 'Y'): ?>
                                            <li><i class="bi bi-person-fill me-2"></i> Child Seat Available</li>
                                        <?php endif; ?>
                                        <?php if ($row['wide_leg_room'] == 'Y'): ?>
                                            <li><i class="bi bi-arrows-angle-expand text-primary me-2"></i> Wide Leg Room</li>
                                        <?php endif; ?>
                                        <?php if ($row['terrain']): ?>
                                            <li><i class="bi bi-signpost-2-fill text-success me-2"></i> <?= $row['terrain'] ?> Terrain</li>
                                        <?php endif; ?>
                                        <?php if ($row['budget_friendly'] == 'Y'): ?>
                                            <li><i class="bi bi-currency-exchange text-success me-2"></i> Budget Friendly</li>
                                        <?php endif; ?>
                                        <?php if ($row['aircon'] == 'Y'): ?>
                                            <li><i class="bi bi-snow text-info me-2"></i> Air Conditioned</li>
                                        <?php endif; ?>
                                        <?php if ($row['wide_compartment'] == 'Y'): ?>
                                            <li><i class="bi bi-box-seam-fill text-secondary me-2"></i> Wide Compartment</li>
                                        <?php endif; ?>
                                        <li><i class="bi bi-person-wheelchair text-dark me-2"></i> <?= $row['with_driver'] ? 'With Driver' : 'Self-Drive' ?></li>
                                    </ul>
                                    <div class="d-grid mt-auto gap-2">
                                        <button class="btn btn-primary btn-lg book-now-btn" data-car-id="<?= $row['id'] ?>">Book Now</button>
                                        <button type="button" class="btn btn-outline-primary view-reviews-btn"
                                            data-bs-toggle="modal" data-bs-target="#reviewsModal"
                                            data-car-id="<?= $row['id'] ?>" data-car-name="<?= htmlspecialchars($row['car_name']) ?>">
                                            View Reviews
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    <?php endwhile;
                else: ?>
                    <div class="col-12">
                        <div class="alert alert-info text-center">No cars available at the moment.</div>
                    </div>
                <?php endif; ?>
            </div>

            <!-- Pagination -->
            <nav class="mt-5" aria-label="Page navigation" id="defaultPagination">
                <?php if ($total_pages > 1): ?>
                    <ul class="pagination justify-content-center">
                        <li class="page-item <?= $current_page <= 1 ? 'disabled' : '' ?>">
                            <a class="page-link" href="?page=<?= $current_page - 1 ?>" aria-label="Previous"><span aria-hidden="true">Previous</span></a>
                        </li>
                        <?php
                        $start = max(1, $current_page - 2);
                        $end = min($total_pages, $current_page + 2);
                        if ($start > 1): ?>
                            <li class="page-item"><a class="page-link" href="?page=1">1</a></li>
                            <?php if ($start > 2): ?><li class="page-item disabled"><span class="page-link">...</span></li><?php endif; ?>
                        <?php endif;
                        for ($i = $start; $i <= $end; $i++): ?>
                            <li class="page-item <?= $i == $current_page ? 'active' : '' ?>">
                                <a class="page-link" href="?page=<?= $i ?>"><?= $i ?></a>
                            </li>
                        <?php endfor;
                        if ($end < $total_pages): ?>
                            <?php if ($end < $total_pages - 1): ?><li class="page-item disabled"><span class="page-link">...</span></li><?php endif; ?>
                            <li class="page-item"><a class="page-link" href="?page=<?= $total_pages ?>"><?= $total_pages ?></a></li>
                        <?php endif; ?>
                        <li class="page-item <?= $current_page >= $total_pages ? 'disabled' : '' ?>">
                            <a class="page-link" href="?page=<?= $current_page + 1 ?>" aria-label="Next"><span aria-hidden="true">Next</span></a>
                        </li>
                    </ul>
                <?php endif; ?>
            </nav>
        </div>
    </section>

    <!-- Reviews Modal -->
    <div class="modal fade" id="reviewsModal" tabindex="-1" aria-labelledby="reviewsModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="reviewsModalLabel">Reviews for <span id="modalCarName"></span></h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div id="reviewsContent">
                        <p>Loading reviews...</p>
                    </div>
                </div>
                <div class="modal-footer"><button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button></div>
            </div>
        </div>
    </div>
    <script>
        /* --------------------------------------------------------------------- */
        /* CAPACITY MAPPING (unchanged) */
        /* --------------------------------------------------------------------- */
        const capacityMapping = {
            'Sedan': ['Small'],
            'SUV': ['Medium'],
            'MPV': ['Medium'],
            'Van': ['Medium Group', 'Large Group'],
            'Mini Truck': ['Small Group', 'Large Group']
        };
        document.getElementById('car_type').addEventListener('change', function() {
            const type = this.value;
            const capSelect = document.getElementById('capacity');
            capSelect.innerHTML = '<option value="">Auto</option>';
            if (capacityMapping[type]) {
                capacityMapping[type].forEach(cap => {
                    const opt = document.createElement('option');
                    opt.value = cap;
                    opt.textContent = cap;
                    capSelect.appendChild(opt);
                });
                capSelect.value = capacityMapping[type][0];
            }
        });

        /* --------------------------------------------------------------------- */
        /* RESET / FIND CARS (unchanged) */
        /* --------------------------------------------------------------------- */
        document.getElementById('resetFilters').onclick = () => location.reload();

        document.getElementById('findCarsForm').onsubmit = async (e) => {
            e.preventDefault();
            const form = new FormData(e.target);
            const payload = {
                car_type: form.get('car_type') || '',
                capacity: form.get('capacity') || '',
                duration: form.get('duration') || '',
                budget: parseFloat(form.get('budget')) || 0,
                inquiry: (form.get('inquiry') || '').trim().toLowerCase()
            };
            try {
                const res = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) throw new Error('API error');
                const data = await res.json();
                renderCars(data.cars, data.recommended_type, data.why_recommended, payload);
            } catch (err) {
                console.warn('ML API failed → fallback to PHP', err);
                const params = new URLSearchParams(payload);
                const phpRes = await fetch(`get_recommendations.php?${params}`);
                const html = await phpRes.text();
                document.getElementById('carContainer').innerHTML = html;
                hidePagination();
            }
        };

        /* --------------------------------------------------------------------- */
        /* RENDER CARS – MODAL + PRICE + MATCHING */
        /* --------------------------------------------------------------------- */
        function renderCars(cars, recommendedType, why, userInputs) {
            const container = document.getElementById('carContainer');
            hidePagination();

            if (!cars || cars.length === 0) {
                container.innerHTML = `<div class="col-12"><div class="alert alert-info text-center">
            No ${recommendedType || ''} cars available right now.</div></div>`;
                return;
            }

            /* ---------- MODAL (identical to PHP) ---------- */
            const modalHTML = cars.map(row => {
                const whyText = why && why[row.id] ? why[row.id] : '';
                const isMatch = whyText.includes('Matches') || whyText.includes('preference') ||
                    whyText.includes('Color') || whyText.includes('Wide') || whyText.includes('Driver');

                return `
        <div class="modal fade" id="bookingModal${row.id}" tabindex="-1"
             aria-labelledby="bookingModalLabel${row.id}" aria-hidden="true">
            <div class="modal-dialog"><div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="bookingModalLabel${row.id}">Book ${escape(row.car_name)}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form method="post" action="">
                        <input type="hidden" name="car_id" value="${row.id}">
                        <input type="hidden" name="duration" value="12hrs">

                        <!-- ALL RATE FIELDS (exactly as PHP) -->
                        <input type="hidden" id="rate_inside_12_${row.id}" value="${row.rate_inside_zambales_12hrs ?? 3000}">
                        <input type="hidden" id="rate_inside_24_${row.id}" value="${row.rate_inside_zambales_24hrs ?? 5000}">
                        <input type="hidden" id="rate_outside_12_${row.id}" value="${row.rate_outside_zambales_12hrs ?? 3500}">
                        <input type="hidden" id="rate_outside_24_${row.id}" value="${row.rate_outside_zambales_24hrs ?? 6000}">
                        <input type="hidden" id="rate_baguio_12_${row.id}" value="${row.rate_baguio_12hrs ?? 4000}">
                        <input type="hidden" id="rate_baguio_24_${row.id}" value="${row.rate_baguio_24hrs ?? 7000}">
                        <input type="hidden" id="driver_price_12_${row.id}" value="${row.driver_price_12hrs ?? 500}">
                        <input type="hidden" id="driver_price_24_${row.id}" value="${row.driver_price_24hrs ?? 800}">
                        <input type="hidden" id="with_driver_${row.id}" value="${row.with_driver ? 'true' : 'false'}">
                        <input type="hidden" id="stakeholder_id_${row.id}" value="${row.stakeholder_id}">

                        <div class="mb-3"><label class="form-label">Full Name</label>
                            <input type="text" class="form-control" value="<?= htmlspecialchars($user_name) ?>" disabled>
                        </div>
                        <div class="mb-3"><label for="address${row.id}" class="form-label">Address</label>
                            <input type="text" class="form-control" id="address${row.id}" name="address" required>
                        </div>
                        <div class="mb-3"><label for="destination${row.id}" class="form-label">Destination</label>
                            <select class="form-select" id="destination${row.id}" name="destination" required onchange="updateTotalPrice(${row.id})">
                                <option value="">Select Destination</option>
                                ${['Inside Zambales','Outside Zambales','Baguio'].map(d=>`<option value="${d}">${d}</option>`).join('')}
                            </select>
                        </div>

                        <div class="mb-3"><label class="form-label">Booking Status</label>
                            <input type="text" class="form-control" value="Pending" disabled>
                        </div>
                        <div class="mb-3"><label class="form-label">Payment Status</label>
                            <input type="text" class="form-control" value="Unpaid" disabled>
                        </div>

                        <div class="mb-3"><label for="booking_date${row.id}" class="form-label">Booking Date & Time</label>
                            <input type="datetime-local" class="form-control" id="booking_date${row.id}"
                                   name="booking_date" required min="${new Date().toISOString().slice(0,16)}"
                                   value="${new Date().toISOString().slice(0,16)}"
                                   onchange="updateTotalPrice(${row.id}); validateBookingDate(${row.id})">
                            <small id="booking_date_error_${row.id}" class="text-danger" style="display:none;">Booking date must be in the future.</small>
                        </div>

                        <div class="mb-3"><label for="return_date${row.id}" class="form-label">Return Date & Time</label>
                            <input type="datetime-local" class="form-control" id="return_date${row.id}"
                                   name="return_date" required min="${new Date().toISOString().slice(0,16)}"
                                   onchange="updateTotalPrice(${row.id}); validateReturnDate(${row.id})">
                            <small id="return_date_error_${row.id}" class="text-danger" style="display:none;">Return date must be after booking date.</small>
                            <small id="mk_rental_error_${row.id}" class="text-danger" style="display:none;">MK Car Rental requires minimum 48 hours.</small>
                        </div>

                        <div class="mb-3"><label class="form-label">Total Price</label>
                            <input type="text" class="form-control" id="total_price${row.id}" value="₱0.00" disabled>
                        </div>

                        <div class="mb-3"><small class="text-muted">Overstay by 1 hour = extra full rate.</small>
                            ${row.stakeholder_id == 7 ? '<br><small class="text-muted">MK Car Rental: Minimum 2 days (48 hours).</small>' : ''}
                        </div>

                        <button type="submit" name="book_car" class="btn btn-primary w-100" id="submit_btn_${row.id}">Confirm Booking</button>
                    </form>
                </div>
            </div></div>
        </div>`;
            }).join('');

            /* ---------- CAR CARD ---------- */
            const cardHTML = cars.map(row => {
                const base12 = row.rate_inside_zambales_12hrs ?? 3000;
                const driver12 = row.with_driver ? (row.driver_price_12hrs ?? 0) : 0;
                const total = base12 + driver12;
                const img = row.car_image ? `../stakeholders/uploads/cars/${row.car_image}` : '../stakeholders/uploads/cars/PLACEHOLDER.png';
                const whyText = why && why[row.id] ? why[row.id] : '';
                const isMatch = !!whyText; // any why → match

                return `
        <div class="col-md-6 col-lg-4">
            <div class="card car-card h-100 shadow-sm ${isMatch ? 'border-success border-2' : ''}">
                ${isMatch ? '<div class="position-absolute top-0 start-0 p-2"><span class="badge bg-success">MATCH</span></div>' : ''}
                <div class="position-relative">
                    <img src="${img}" class="car-img-top" alt="${escape(row.car_name)}">
                </div>
                <div class="card-body d-flex flex-column">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="car-title mb-0">${escape(row.car_name)}</h5>
                        <span class="badge bg-secondary">${escape(row.shop_name)}</span>
                    </div>
                    <div class="d-flex justify-content-end align-items-center mb-3">
                        <span class="car-price">
                            ₱${Number(total).toLocaleString('en-US',{minimumFractionDigits:2})}
                            <small class="text-muted">/12hrs</small>
                            ${row.with_driver && driver12>0 ? `<br><small class="text-success">(Incl. ₱${Number(driver12).toLocaleString()} driver)</small>` : ''}
                        </span>
                    </div>
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <small class="text-muted">
                            Booked <strong>${row.booking_count}</strong> time${row.booking_count>1?'s':''}
                        </small>
                    </div>

                    <ul class="car-features list-unstyled mb-4">
                        <li><i class="bi bi-people-fill text-primary me-2"></i> ${row.capacity} Seats</li>
                        <li><i class="bi bi-droplet-fill text-info me-2"></i> ${row.color||'N/A'}</li>
                        <li><i class="bi bi-gear-fill text-secondary me-2"></i> ${row.transmission}</li>
                        <li><i class="bi bi-fuel-pump-fill text-success me-2"></i> ${row.fuel_type}</li>
                        <li><i class="bi bi-car-front-fill text-indigo me-2"></i> ${row.car_type}</li>
                        ${row.special_needs_friendly==='Y' ? '<li><i class="bi bi-universal-access-circle-fill text-warning me-2"></i> PWD & Senior Friendly</li>' : ''}
                        ${row.child_seat==='Y' ? '<li><i class="bi bi-person-fill me-2"></i> Child Seat</li>' : ''}
                        ${row.wide_leg_room==='Y' ? '<li><i class="bi bi-arrows-angle-expand text-primary me-2"></i> Wide Leg Room</li>' : ''}
                        ${row.terrain ? `<li><i class="bi bi-signpost-2-fill text-success me-2"></i> ${row.terrain} Terrain</li>` : ''}
                        ${row.budget_friendly==='Y' ? '<li><i class="bi bi-currency-exchange text-success me-2"></i> Budget Friendly</li>' : ''}
                        ${row.aircon==='Y' ? '<li><i class="bi bi-snow text-info me-2"></i> Air Conditioned</li>' : ''}
                        ${row.wide_compartment==='Y' ? '<li><i class="bi bi-box-seam-fill text-secondary me-2"></i> Wide Compartment</li>' : ''}
                        <li><i class="bi bi-person-wheelchair text-dark me-2"></i> ${row.with_driver?'With Driver':'Self-Drive'}</li>
                    </ul>

                    ${whyText ? `
                    <div class="why-recommended mt-3 p-2 rounded bg-light border-start border-success border-3">
                        <i class="bi bi-check-circle-fill text-success me-1"></i>
                        <strong>Why recommended:</strong><br>${whyText}
                    </div>` : ''}

                    <div class="d-grid mt-auto gap-2">
                        <button class="btn btn-primary btn-lg book-now-btn" data-car-id="${row.id}">Book Now</button>
                        <button type="button" class="btn btn-outline-primary view-reviews-btn"
                                data-bs-toggle="modal" data-bs-target="#reviewsModal"
                                data-car-id="${row.id}" data-car-name="${escape(row.car_name)}">View Reviews</button>
                    </div>
                </div>
            </div>
        </div>`;
            }).join('');

            container.innerHTML = modalHTML + cardHTML;
        }

        /* --------------------------------------------------------------------- */
        /* UTILS */
        /* --------------------------------------------------------------------- */
        function hidePagination() {
            const n = document.getElementById('defaultPagination');
            if (n) n.style.display = 'none';
        }

        function escape(s) {
            return s ? document.createElement('div').appendChild(document.createTextNode(s)).parentNode.innerHTML : '';
        }

        /* --------------------------------------------------------------------- */
        /* BOOK NOW → OPEN MODAL */
        /* --------------------------------------------------------------------- */
        document.addEventListener('click', e => {
            if (e.target && e.target.classList.contains('book-now-btn')) {
                const id = e.target.getAttribute('data-car-id');
                const modal = document.getElementById('bookingModal' + id);
                if (modal) {
                    new bootstrap.Modal(modal).show();
                    setTimeout(() => updateTotalPrice(id), 200);
                }
            }
        });

        /* --------------------------------------------------------------------- */
        /* REVIEWS MODAL (unchanged) */
        /* --------------------------------------------------------------------- */
        $(document).on('click', '.view-reviews-btn', function() {
            const carId = $(this).data('car-id'),
                carName = $(this).data('car-name');
            $('#modalCarName').text(carName);
            $.post('get_reviews.php', {
                    car_id: carId
                }, html => $('#reviewsContent').html(html))
                .fail(() => $('#reviewsContent').html('<p class="text-danger">Error loading reviews.</p>'));
        });

        /* --------------------------------------------------------------------- */
        /* PRICE CALCULATION – 12/24 hrs, any destination, driver, MK rule */
        /* --------------------------------------------------------------------- */
        function updateTotalPrice(carId) {
            const dest = document.getElementById(`destination${carId}`)?.value;
            const book = document.getElementById(`booking_date${carId}`)?.value;
            const ret = document.getElementById(`return_date${carId}`)?.value;
            if (!dest || !book || !ret) {
                document.getElementById(`total_price${carId}`).value = '₱0.00';
                return;
            }

            const b = new Date(book),
                r = new Date(ret);
            if (r <= b) {
                document.getElementById(`total_price${carId}`).value = 'Invalid';
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

            // ----- choose correct base & driver price (12h vs 24h) -----
            const is24h = document.querySelector(`input[name="duration"][value="24hrs"]`)?.checked; // if you add 24h radio later
            const baseKey = dest === 'Baguio' ? (is24h ? 'baguio_24' : 'baguio_12') :
                dest === 'Inside Zambales' ? (is24h ? 'inside_24' : 'inside_12') :
                (is24h ? 'outside_24' : 'outside_12');
            const base = rates[baseKey];
            const driver = is24h ? rates.driver_24 : rates.driver_12;

            const hours = (r - b) / 36e5;
            const mult = Math.ceil(hours / (is24h ? 24 : 12));
            const total = (base * mult) + (rates.withDriver ? driver * mult : 0);

            document.getElementById(`total_price${carId}`).value = '₱' + total.toLocaleString('en-US', {
                minimumFractionDigits: 2
            });
        }

        /* --------------------------------------------------------------------- */
        /* VALIDATION (same as PHP) */
        /* --------------------------------------------------------------------- */
        function validateBookingDate(carId) {
            const input = document.getElementById(`booking_date${carId}`);
            const sel = new Date(input.value);
            const now = new Date();
            const err = document.getElementById(`booking_date_error_${carId}`);
            const btn = document.getElementById(`submit_btn_${carId}`);
            if (sel < now) {
                err.style.display = 'block';
                btn.disabled = true;
            } else {
                err.style.display = 'none';
                validateReturnDate(carId);
            }
        }

        function validateReturnDate(carId) {
            const b = document.getElementById(`booking_date${carId}`).value;
            const r = document.getElementById(`return_date${carId}`).value;
            if (!b || !r) return;
            const bd = new Date(b),
                rd = new Date(r);
            const hours = (rd - bd) / 36e5;
            const errR = document.getElementById(`return_date_error_${carId}`);
            const errM = document.getElementById(`mk_rental_error_${carId}`);
            const btn = document.getElementById(`submit_btn_${carId}`);
            let ok = true;
            if (rd <= bd) {
                errR.style.display = 'block';
                ok = false;
            } else errR.style.display = 'none';
            if (document.getElementById(`stakeholder_id_${carId}`).value == 7 && hours < 48) {
                errM.style.display = 'block';
                ok = false;
            } else errM.style.display = 'none';
            btn.disabled = !ok;
        }
    </script>
 